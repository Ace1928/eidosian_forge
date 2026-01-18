import pytest
from spacy.strings import StringStore
@pytest.mark.parametrize('text', ['qqqqq'])
def test_stringstore_to_bytes(stringstore, text):
    store = stringstore.add(text)
    serialized = stringstore.to_bytes()
    new_stringstore = StringStore().from_bytes(serialized)
    assert new_stringstore[store] == text