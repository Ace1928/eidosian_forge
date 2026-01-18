import pytest
from spacy.lang.ta import Tamil
from spacy.symbols import ORTH
@pytest.mark.parametrize('text,expected_tokens', TA_BASIC_TOKENIZATION_TESTS)
def test_ta_tokenizer_basic(ta_tokenizer, text, expected_tokens):
    tokens = ta_tokenizer(text)
    token_list = [token.text for token in tokens]
    assert expected_tokens == token_list