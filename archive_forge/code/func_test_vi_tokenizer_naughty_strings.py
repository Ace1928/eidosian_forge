import pytest
from spacy.lang.vi import Vietnamese
from ...tokenizer.test_naughty_strings import NAUGHTY_STRINGS
@pytest.mark.parametrize('text', NAUGHTY_STRINGS)
def test_vi_tokenizer_naughty_strings(vi_tokenizer, text):
    tokens = vi_tokenizer(text)
    assert tokens.text_with_ws == text