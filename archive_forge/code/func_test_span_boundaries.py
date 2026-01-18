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
def test_span_boundaries(doc):
    start = 1
    end = 5
    span = doc[start:end]
    for i in range(start, end):
        assert span[i - start] == doc[i]
    with pytest.raises(IndexError):
        span[-5]
    with pytest.raises(IndexError):
        span[5]
    empty_span_0 = doc[0:0]
    assert empty_span_0.text == ''
    assert empty_span_0.start == 0
    assert empty_span_0.end == 0
    assert empty_span_0.start_char == 0
    assert empty_span_0.end_char == 0
    empty_span_1 = doc[1:1]
    assert empty_span_1.text == ''
    assert empty_span_1.start == 1
    assert empty_span_1.end == 1
    assert empty_span_1.start_char == empty_span_1.end_char
    oob_span_start = doc[-len(doc) - 1:-len(doc) - 10]
    assert oob_span_start.text == ''
    assert oob_span_start.start == 0
    assert oob_span_start.end == 0
    assert oob_span_start.start_char == 0
    assert oob_span_start.end_char == 0
    oob_span_end = doc[len(doc) + 1:len(doc) + 10]
    assert oob_span_end.text == ''
    assert oob_span_end.start == len(doc)
    assert oob_span_end.end == len(doc)
    assert oob_span_end.start_char == len(doc.text)
    assert oob_span_end.end_char == len(doc.text)