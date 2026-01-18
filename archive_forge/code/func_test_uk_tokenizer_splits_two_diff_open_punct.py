import pytest
@pytest.mark.parametrize('punct', PUNCT_OPEN)
@pytest.mark.parametrize('punct_add', ['`'])
@pytest.mark.parametrize('text', ['Привет', 'Привіт', 'Ґелґотати', "З'єднання", 'Єдність', 'їхні'])
def test_uk_tokenizer_splits_two_diff_open_punct(uk_tokenizer, punct, punct_add, text):
    tokens = uk_tokenizer(punct + punct_add + text)
    assert len(tokens) == 3
    assert tokens[0].text == punct
    assert tokens[1].text == punct_add
    assert tokens[2].text == text