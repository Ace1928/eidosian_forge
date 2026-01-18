import pytest
def test_token_morph_eq(i_has):
    assert i_has[0].morph is not i_has[0].morph
    assert i_has[0].morph == i_has[0].morph
    assert i_has[0].morph != i_has[1].morph