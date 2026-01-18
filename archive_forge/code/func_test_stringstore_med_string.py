import pytest
from spacy.strings import StringStore
@pytest.mark.parametrize('text1,text2', [(b'0123456789', b'A')])
def test_stringstore_med_string(stringstore, text1, text2):
    store = stringstore.add(text1)
    assert stringstore[store] == text1.decode('utf8')
    stringstore.add(text2)
    assert stringstore[text1] == store