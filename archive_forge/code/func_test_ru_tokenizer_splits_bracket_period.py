from string import punctuation
import pytest
def test_ru_tokenizer_splits_bracket_period(ru_tokenizer):
    text = '(Раз, два, три, проверка).'
    tokens = ru_tokenizer(text)
    assert tokens[len(tokens) - 1].text == '.'