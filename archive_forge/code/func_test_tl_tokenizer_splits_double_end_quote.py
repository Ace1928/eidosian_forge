import pytest
from spacy.lang.punctuation import TOKENIZER_PREFIXES
from spacy.util import compile_prefix_regex
@pytest.mark.parametrize('text', ["Mabuhay''"])
def test_tl_tokenizer_splits_double_end_quote(tl_tokenizer, text):
    tokens = tl_tokenizer(text)
    assert len(tokens) == 2
    tokens_punct = tl_tokenizer("''")
    assert len(tokens_punct) == 1