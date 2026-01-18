import pytest
@pytest.mark.parametrize('punct', PUNCT_CLOSE)
@pytest.mark.parametrize('text', ['Здраво'])
def test_sr_tokenizer_splits_close_punct(sr_tokenizer, punct, text):
    tokens = sr_tokenizer(text + punct)
    assert len(tokens) == 2
    assert tokens[0].text == text
    assert tokens[1].text == punct