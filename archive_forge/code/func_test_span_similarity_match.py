import numpy
import pytest
from numpy.testing import assert_array_equal
from thinc.api import get_current_ops
from spacy.attrs import LENGTH, ORTH
from spacy.lang.en import English
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans
from spacy.vocab import Vocab
from ..util import add_vecs_to_vocab
from .test_underscore import clean_underscore  # noqa: F401
def test_span_similarity_match():
    doc = Doc(Vocab(), words=['a', 'b', 'a', 'b'])
    span1 = doc[:2]
    span2 = doc[2:]
    with pytest.warns(UserWarning):
        assert span1.similarity(span2) == 1.0
        assert span1.similarity(doc) == 0.0
        assert span1[:1].similarity(doc.vocab['a']) == 1.0