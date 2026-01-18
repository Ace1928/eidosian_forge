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
def test_span_string_label_id(doc):
    span = Span(doc, 0, 1, label='hello', span_id='Q342')
    assert span.label_ == 'hello'
    assert span.label == doc.vocab.strings['hello']
    assert span.id_ == 'Q342'
    assert span.id == doc.vocab.strings['Q342']