import pytest
from hypothesis import settings
from spacy.util import get_lang_class
@pytest.fixture(scope='session')
def vi_tokenizer():
    pytest.importorskip('pyvi')
    return get_lang_class('vi')().tokenizer