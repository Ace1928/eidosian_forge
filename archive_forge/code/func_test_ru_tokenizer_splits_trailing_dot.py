from string import punctuation
import pytest
@pytest.mark.parametrize('text', ['Тест.'])
def test_ru_tokenizer_splits_trailing_dot(ru_tokenizer, text):
    tokens = ru_tokenizer(text)
    assert tokens[1].text == '.'