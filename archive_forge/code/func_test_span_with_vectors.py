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
def test_span_with_vectors(doc):
    ops = get_current_ops()
    prev_vectors = doc.vocab.vectors
    vectors = [('apple', ops.asarray([1, 2, 3])), ('orange', ops.asarray([-1, -2, -3])), ('And', ops.asarray([-1, -1, -1])), ('juice', ops.asarray([5, 5, 10])), ('pie', ops.asarray([7, 6.3, 8.9]))]
    add_vecs_to_vocab(doc.vocab, vectors)
    assert_array_equal(ops.to_numpy(doc[0:0].vector), numpy.zeros((3,)))
    assert_array_equal(ops.to_numpy(doc[0:4].vector), numpy.zeros((3,)))
    assert_array_equal(ops.to_numpy(doc[10:11].vector), [-1, -1, -1])
    doc.vocab.vectors = prev_vectors