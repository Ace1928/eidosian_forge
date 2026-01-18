import pytest
from spacy.lang.punctuation import TOKENIZER_PREFIXES
from spacy.util import compile_prefix_regex
@pytest.mark.parametrize('punct_open,punct_close', PUNCT_PAIRED)
@pytest.mark.parametrize('punct_open2,punct_close2', [('`', "'")])
@pytest.mark.parametrize('text', ['Mabuhay'])
def test_tl_tokenizer_two_diff_punct(tl_tokenizer, punct_open, punct_close, punct_open2, punct_close2, text):
    tokens = tl_tokenizer(punct_open2 + punct_open + text + punct_close + punct_close2)
    assert len(tokens) == 5
    assert tokens[0].text == punct_open2
    assert tokens[1].text == punct_open
    assert tokens[2].text == text
    assert tokens[3].text == punct_close
    assert tokens[4].text == punct_close2