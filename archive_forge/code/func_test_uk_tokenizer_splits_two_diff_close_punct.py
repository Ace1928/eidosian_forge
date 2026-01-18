import pytest
@pytest.mark.parametrize('punct', PUNCT_CLOSE)
@pytest.mark.parametrize('punct_add', ["'"])
@pytest.mark.parametrize('text', ['Привет', 'Привіт', 'Ґелґотати', "З'єднання", 'Єдність', 'їхні'])
def test_uk_tokenizer_splits_two_diff_close_punct(uk_tokenizer, punct, punct_add, text):
    tokens = uk_tokenizer(text + punct + punct_add)
    assert len(tokens) == 3
    assert tokens[0].text == text
    assert tokens[1].text == punct
    assert tokens[2].text == punct_add