import pickle
from spacy.lang.th import Thai
from ...util import make_tempdir
def test_th_tokenizer_pickle(th_tokenizer):
    b = pickle.dumps(th_tokenizer)
    th_tokenizer_re = pickle.loads(b)
    assert th_tokenizer.to_bytes() == th_tokenizer_re.to_bytes()