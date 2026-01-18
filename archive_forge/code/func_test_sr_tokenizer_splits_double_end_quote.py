import pytest
@pytest.mark.parametrize('text', ["Тест''"])
def test_sr_tokenizer_splits_double_end_quote(sr_tokenizer, text):
    tokens = sr_tokenizer(text)
    assert len(tokens) == 2
    tokens_punct = sr_tokenizer("''")
    assert len(tokens_punct) == 1