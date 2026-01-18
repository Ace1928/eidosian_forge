from string import punctuation
import pytest
@pytest.mark.parametrize('text', ["Тест''"])
def test_ru_tokenizer_splits_double_end_quote(ru_tokenizer, text):
    tokens = ru_tokenizer(text)
    assert len(tokens) == 2
    tokens_punct = ru_tokenizer("''")
    assert len(tokens_punct) == 1