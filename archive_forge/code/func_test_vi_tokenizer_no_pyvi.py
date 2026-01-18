import pytest
from spacy.lang.vi import Vietnamese
from ...tokenizer.test_naughty_strings import NAUGHTY_STRINGS
def test_vi_tokenizer_no_pyvi():
    """Test for whitespace tokenization without pyvi"""
    nlp = Vietnamese.from_config({'nlp': {'tokenizer': {'use_pyvi': False}}})
    text = 'Đây là một văn  bản bằng tiếng Việt Sau đó, đây là một văn bản khác bằng ngôn ngữ này'
    doc = nlp(text)
    assert [t.text for t in doc if not t.is_space] == text.split()
    assert doc[4].text == ' '