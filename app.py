import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Study Notes Search", page_icon="📚")
st.title("Study Notes Search")
st.write("Find relevant passages in your notes using TF-IDF text retrieval.")
st.caption("A classical machine-learning retrieval prototype, not a generative AI chatbot.")

DEMO = """Computer networks connect devices so they can exchange data. TCP provides reliable, ordered delivery using acknowledgments and retransmissions.

UDP sends datagrams without guaranteeing delivery or ordering. Applications may choose UDP when low latency is more important than retransmission.

A relational database stores data in tables. SQL is used to query and manipulate data. A primary key uniquely identifies a row.

A database index can speed up reads but requires storage and can add overhead to writes.

Supervised learning trains a model using labeled examples. Classification predicts categories while regression predicts numerical values.

TF-IDF represents documents using term frequency and inverse document frequency. Cosine similarity compares the directions of document vectors."""

def passages(text):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    result = []
    for paragraph in paragraphs:
        words = paragraph.split()
        for start in range(0, len(words), 100):
            result.append(" ".join(words[start:start + 100]))
    return result

def search(text, question, limit=3):
    chunks = passages(text)
    if not chunks:
        return []
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        matrix = vectorizer.fit_transform(chunks)
    except ValueError:
        return []
    query = vectorizer.transform([question])
    scores = cosine_similarity(query, matrix).ravel()
    order = scores.argsort()[::-1]
    return [(int(i) + 1, chunks[i], float(scores[i])) for i in order[:limit] if scores[i] > 0]

notes = st.text_area("Paste study notes (blank lines separate passages)", DEMO, height=260, max_chars=60000)
question = st.text_input("Search question", "How does TCP provide reliable delivery?")
limit = st.slider("Maximum results", 1, 5, 3)
if st.button("Search notes", type="primary"):
    if not notes.strip() or not question.strip():
        st.warning("Enter both notes and a question.")
    else:
        results = search(notes, question, limit)
        if not results:
            st.info("No matching terms found. Try words used in your notes. This app does not invent answers.")
        else:
            export = []
            for number, (source, passage, score) in enumerate(results, 1):
                st.subheader(f"Result {number} — passage {source}")
                st.write(passage)
                st.caption(f"Cosine similarity: {score:.3f}. This is a ranking score, not a confidence probability.")
                export.append(f"Passage {source} | similarity {score:.3f}\n{passage}")
            st.download_button("Download results", "\n\n".join(export), "search-results.txt", "text/plain")
with st.expander("How it works and limitations"):
    st.write("The app divides notes into passages of up to 100 words, computes TF-IDF features, and ranks passages using cosine similarity. It returns original passages rather than generating answers. It can miss synonyms, does not understand meaning like an LLM, and does not guarantee that a positive match answers your question. Notes are processed by the running app; this code does not deliberately save them or send them to an external model API. Do not upload sensitive material to an untrusted deployment.")
