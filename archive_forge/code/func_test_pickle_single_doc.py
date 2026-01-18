from spacy.compat import pickle
from spacy.language import Language
def test_pickle_single_doc():
    nlp = Language()
    doc = nlp('pickle roundtrip')
    data = pickle.dumps(doc, 1)
    doc2 = pickle.loads(data)
    assert doc2.text == 'pickle roundtrip'