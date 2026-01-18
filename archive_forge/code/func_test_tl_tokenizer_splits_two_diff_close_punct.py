import pytest
from spacy.lang.punctuation import TOKENIZER_PREFIXES
from spacy.util import compile_prefix_regex
@pytest.mark.parametrize('punct', PUNCT_CLOSE)
@pytest.mark.parametrize('punct_add', ['`'])
@pytest.mark.parametrize('text', ['Mabuhay'])
def test_tl_tokenizer_splits_two_diff_close_punct(tl_tokenizer, punct, punct_add, text):
    tokens = tl_tokenizer(text + punct + punct_add)
    assert len(tokens) == 3
    assert tokens[0].text == text
    assert tokens[1].text == punct
    assert tokens[2].text == punct_add