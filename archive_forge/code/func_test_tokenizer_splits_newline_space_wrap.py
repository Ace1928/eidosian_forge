import pytest
@pytest.mark.parametrize('text', ['lorem \n ipsum'])
def test_tokenizer_splits_newline_space_wrap(tokenizer, text):
    tokens = tokenizer(text)
    assert len(tokens) == 3