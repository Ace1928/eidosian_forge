import pytest
@pytest.mark.parametrize('text,norms', NORM_TESTCASES)
def test_tt_tokenizer_handles_norm_exceptions(tt_tokenizer, text, norms):
    tokens = tt_tokenizer(text)
    assert [token.norm_ for token in tokens] == norms