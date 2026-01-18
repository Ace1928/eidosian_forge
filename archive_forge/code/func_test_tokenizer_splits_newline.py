import pytest
@pytest.mark.parametrize('text', ['lorem\nipsum'])
def test_tokenizer_splits_newline(tokenizer, text):
    tokens = tokenizer(text)
    assert len(tokens) == 3
    assert tokens[1].text == '\n'