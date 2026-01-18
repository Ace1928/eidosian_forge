import pytest
@pytest.mark.parametrize('text,expected_tokens', SV_TOKEN_EXCEPTION_TESTS)
def test_sv_tokenizer_handles_exception_cases(sv_tokenizer, text, expected_tokens):
    tokens = sv_tokenizer(text)
    token_list = [token.text for token in tokens if not token.is_space]
    assert expected_tokens == token_list