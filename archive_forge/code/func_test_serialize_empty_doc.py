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
def test_serialize_empty_doc(en_vocab):
    doc = Doc(en_vocab)
    data = doc.to_bytes()
    doc2 = Doc(en_vocab)
    doc2.from_bytes(data)
    assert len(doc) == len(doc2)
    for token1, token2 in zip(doc, doc2):
        assert token1.text == token2.text