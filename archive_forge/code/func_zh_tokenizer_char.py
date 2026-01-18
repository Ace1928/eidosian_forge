import pytest
from hypothesis import settings
from spacy.util import get_lang_class
@pytest.fixture(scope='session')
def zh_tokenizer_char():
    nlp = get_lang_class('zh')()
    return nlp.tokenizer