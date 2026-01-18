import pytest
@pytest.mark.parametrize('text', ["(Ma'arif?)"])
def test_tokenizer_splits_uneven_wrap(id_tokenizer, text):
    tokens = id_tokenizer(text)
    assert len(tokens) == 4