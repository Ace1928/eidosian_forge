import pytest
@pytest.mark.parametrize('text,expected_tokens', TESTCASES)
def test_tt_tokenizer_handles_testcases(tt_tokenizer, text, expected_tokens):
    tokens = [token.text for token in tt_tokenizer(text) if not token.is_space]
    assert expected_tokens == tokens