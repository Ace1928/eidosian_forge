import pytest
from spacy.lang.tr.lex_attrs import like_num
@pytest.mark.parametrize('word', ['beş', 'yedi', 'yedinci', 'birinci', 'milyonuncu'])
def test_tr_lex_attrs_capitals(word):
    assert like_num(word)
    assert like_num(word.upper())