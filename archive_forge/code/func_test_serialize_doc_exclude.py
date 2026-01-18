import copy
import pickle
import numpy
import pytest
from spacy.attrs import DEP, HEAD
from spacy.lang.en import English
from spacy.language import Language
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc
from spacy.vectors import Vectors
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_serialize_doc_exclude(en_vocab):
    doc = Doc(en_vocab, words=['hello', 'world'])
    doc.user_data['foo'] = 'bar'
    new_doc = Doc(en_vocab).from_bytes(doc.to_bytes())
    assert new_doc.user_data['foo'] == 'bar'
    new_doc = Doc(en_vocab).from_bytes(doc.to_bytes(), exclude=['user_data'])
    assert not new_doc.user_data
    new_doc = Doc(en_vocab).from_bytes(doc.to_bytes(exclude=['user_data']))
    assert not new_doc.user_data