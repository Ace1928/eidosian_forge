import pytest
def test_sr_tokenizer_splits_bracket_period(sr_tokenizer):
    text = '(Један, два, три, четири, проба).'
    tokens = sr_tokenizer(text)
    assert tokens[len(tokens) - 1].text == '.'