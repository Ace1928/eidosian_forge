from spacy.compat import pickle
from spacy.language import Language
def test_user_data_unpickles():
    nlp = Language()
    doc = nlp('Hello')
    doc.user_data[0, 1] = False
    b = pickle.dumps(doc)
    doc2 = pickle.loads(b)
    assert doc2.user_data[0, 1] is False