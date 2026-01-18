import pytest
from hypothesis import settings
from spacy.util import get_lang_class
@pytest.fixture(scope='session')
def uk_lemmatizer():
    pytest.importorskip('pymorphy3')
    pytest.importorskip('pymorphy3_dicts_uk')
    return get_lang_class('uk')().add_pipe('lemmatizer')