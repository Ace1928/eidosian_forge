from string import punctuation
import pytest
@pytest.mark.parametrize('text', ['(', '((', '<'])
def test_ru_tokenizer_handles_only_punct(ru_tokenizer, text):
    tokens = ru_tokenizer(text)
    assert len(tokens) == len(text)