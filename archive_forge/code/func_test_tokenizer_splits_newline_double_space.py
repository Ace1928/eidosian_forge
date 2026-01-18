import pytest
@pytest.mark.parametrize('text', ['lorem  \nipsum'])
def test_tokenizer_splits_newline_double_space(tokenizer, text):
    tokens = tokenizer(text)
    assert len(tokens) == 3