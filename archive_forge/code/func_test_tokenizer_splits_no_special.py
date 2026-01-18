import pytest
@pytest.mark.parametrize('text', ['(under)'])
def test_tokenizer_splits_no_special(sv_tokenizer, text):
    tokens = sv_tokenizer(text)
    assert len(tokens) == 3