import pytest
@pytest.mark.parametrize('text,expected_tokens', TEST_CASES)
def test_ro_tokenizer_handles_testcases(ro_tokenizer, text, expected_tokens):
    tokens = ro_tokenizer(text)
    token_list = [token.text for token in tokens if not token.is_space]
    assert expected_tokens == token_list