import pytest
@pytest.mark.parametrize('text', ['lorem  ipsum'])
def test_tokenizer_splits_double_space(tokenizer, text):
    tokens = tokenizer(text)
    assert len(tokens) == 3
    assert tokens[1].text == ' '