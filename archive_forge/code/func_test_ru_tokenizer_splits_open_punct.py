from string import punctuation
import pytest
@pytest.mark.parametrize('punct', PUNCT_OPEN)
@pytest.mark.parametrize('text', ['Привет'])
def test_ru_tokenizer_splits_open_punct(ru_tokenizer, punct, text):
    tokens = ru_tokenizer(punct + text)
    assert len(tokens) == 2
    assert tokens[0].text == punct
    assert tokens[1].text == text