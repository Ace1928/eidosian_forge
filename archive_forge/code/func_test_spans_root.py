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
def test_spans_root(doc):
    span = doc[2:4]
    assert len(span) == 2
    assert span.text == 'a sentence'
    assert span.root.text == 'sentence'
    assert span.root.head.text == 'is'