import pytest
from spacy.lang.sv.lex_attrs import like_num
@pytest.mark.parametrize('word', ['elva'])
def test_sv_lex_attrs_capitals(word):
    assert like_num(word)
    assert like_num(word.upper())