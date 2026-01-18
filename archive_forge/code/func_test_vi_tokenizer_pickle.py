import pickle
from spacy.lang.vi import Vietnamese
from ...util import make_tempdir
def test_vi_tokenizer_pickle(vi_tokenizer):
    b = pickle.dumps(vi_tokenizer)
    vi_tokenizer_re = pickle.loads(b)
    assert vi_tokenizer.to_bytes() == vi_tokenizer_re.to_bytes()