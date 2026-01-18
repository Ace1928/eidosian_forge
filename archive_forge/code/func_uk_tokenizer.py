import pytest
from hypothesis import settings
from spacy.util import get_lang_class
@pytest.fixture(scope='session')
def uk_tokenizer():
    pytest.importorskip('pymorphy3')
    return get_lang_class('uk')().tokenizer