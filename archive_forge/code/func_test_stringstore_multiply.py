import pytest
from spacy.strings import StringStore
@pytest.mark.parametrize('factor', [254, 255, 256])
def test_stringstore_multiply(stringstore, factor):
    text = 'a' * factor
    store = stringstore.add(text)
    assert stringstore[store] == text