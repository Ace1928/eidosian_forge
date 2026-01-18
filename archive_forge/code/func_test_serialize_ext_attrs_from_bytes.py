import pytest
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_serialize_ext_attrs_from_bytes(doc_w_attrs):
    doc_b = doc_w_attrs.to_bytes()
    doc = Doc(Vocab()).from_bytes(doc_b)
    assert doc._.has('_test_attr')
    assert doc._._test_attr == 'test'
    assert doc._._test_prop == len(doc.text)
    assert doc._._test_method('test') == f'{len(doc.text)}test'
    assert doc[0]._._test_token == 't0'
    assert doc[1]._._test_token == 't1'
    assert doc[2]._._test_token == 't0'