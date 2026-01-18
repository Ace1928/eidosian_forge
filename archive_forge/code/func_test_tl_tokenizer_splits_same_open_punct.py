import pytest
from spacy.lang.punctuation import TOKENIZER_PREFIXES
from spacy.util import compile_prefix_regex
@pytest.mark.parametrize('punct', PUNCT_OPEN)
@pytest.mark.parametrize('text', ['Mabuhay'])
def test_tl_tokenizer_splits_same_open_punct(tl_tokenizer, punct, text):
    tokens = tl_tokenizer(punct + punct + punct + text)
    assert len(tokens) == 4
    assert tokens[0].text == punct
    assert tokens[3].text == text