import pytest
from spacy.strings import StringStore
@pytest.mark.parametrize('text1,text2,text3', [('Hello', 'goodbye', 'hello')])
def test_stringstore_save_unicode(stringstore, text1, text2, text3):
    key = stringstore.add(text1)
    assert stringstore[text1] == key
    assert stringstore[text2] != key
    assert stringstore[text3] != key