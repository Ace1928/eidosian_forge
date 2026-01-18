import pytest
@pytest.mark.slow
@pytest.mark.parametrize('text', NAUGHTY_STRINGS)
def test_tokenizer_naughty_strings(tokenizer, text):
    tokens = tokenizer(text)
    assert tokens.text_with_ws == text