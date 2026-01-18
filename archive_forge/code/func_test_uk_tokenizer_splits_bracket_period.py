import pytest
def test_uk_tokenizer_splits_bracket_period(uk_tokenizer):
    text = '(Раз, два, три, проверка).'
    tokens = uk_tokenizer(text)
    assert tokens[len(tokens) - 1].text == '.'