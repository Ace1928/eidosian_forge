import pickle
import re
import pytest
from spacy.attrs import ENT_IOB, ENT_TYPE
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.util import (
from ..util import assert_packed_msg_equal, make_tempdir
@pytest.mark.parametrize('text', ['I💜you', 'they’re', '“hello”'])
def test_serialize_tokenizer_roundtrip_bytes(en_tokenizer, text):
    tokenizer = en_tokenizer
    new_tokenizer = load_tokenizer(tokenizer.to_bytes())
    assert_packed_msg_equal(new_tokenizer.to_bytes(), tokenizer.to_bytes())
    assert new_tokenizer.to_bytes() == tokenizer.to_bytes()
    doc1 = tokenizer(text)
    doc2 = new_tokenizer(text)
    assert [token.text for token in doc1] == [token.text for token in doc2]