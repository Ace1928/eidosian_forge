import pytest
@pytest.mark.parametrize('text', ["'Тест"])
def test_sr_tokenizer_splits_open_appostrophe(sr_tokenizer, text):
    tokens = sr_tokenizer(text)
    assert len(tokens) == 2
    assert tokens[0].text == "'"