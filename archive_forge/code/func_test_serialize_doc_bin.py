import pytest
import spacy
from spacy.lang.en import English
from spacy.tokens import Doc, DocBin
from spacy.tokens.underscore import Underscore
def test_serialize_doc_bin():
    doc_bin = DocBin(attrs=['LEMMA', 'ENT_IOB', 'ENT_TYPE', 'NORM', 'ENT_ID'], store_user_data=True)
    texts = ['Some text', 'Lots of texts...', '...']
    cats = {'A': 0.5}
    nlp = English()
    for doc in nlp.pipe(texts):
        doc.cats = cats
        span = doc[0:2]
        span.label_ = 'UNUSUAL_SPAN_LABEL'
        span.id_ = 'UNUSUAL_SPAN_ID'
        span.kb_id_ = 'UNUSUAL_SPAN_KB_ID'
        doc.spans['start'] = [span]
        doc[0].norm_ = 'UNUSUAL_TOKEN_NORM'
        doc[0].ent_id_ = 'UNUSUAL_TOKEN_ENT_ID'
        doc_bin.add(doc)
    bytes_data = doc_bin.to_bytes()
    nlp = spacy.blank('en')
    doc_bin = DocBin().from_bytes(bytes_data)
    reloaded_docs = list(doc_bin.get_docs(nlp.vocab))
    for i, doc in enumerate(reloaded_docs):
        assert doc.text == texts[i]
        assert doc.cats == cats
        assert len(doc.spans) == 1
        assert doc.spans['start'][0].label_ == 'UNUSUAL_SPAN_LABEL'
        assert doc.spans['start'][0].id_ == 'UNUSUAL_SPAN_ID'
        assert doc.spans['start'][0].kb_id_ == 'UNUSUAL_SPAN_KB_ID'
        assert doc[0].norm_ == 'UNUSUAL_TOKEN_NORM'
        assert doc[0].ent_id_ == 'UNUSUAL_TOKEN_ENT_ID'