import pytest
@pytest.mark.parametrize('text', ['Hej,Världen', 'en,två'])
def test_tokenizer_splits_comma_infix(sv_tokenizer, text):
    tokens = sv_tokenizer(text)
    assert len(tokens) == 3
    assert tokens[0].text == text.split(',')[0]
    assert tokens[1].text == ','
    assert tokens[2].text == text.split(',')[1]