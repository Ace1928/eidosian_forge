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
def test_span_to_array(doc):
    span = doc[1:-2]
    arr = span.to_array([ORTH, LENGTH])
    assert arr.shape == (len(span), 2)
    assert arr[0, 0] == span[0].orth
    assert arr[0, 1] == len(span[0])