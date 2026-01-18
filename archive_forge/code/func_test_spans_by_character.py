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
def test_spans_by_character(doc):
    span1 = doc[1:-2]
    span2 = doc.char_span(span1.start_char, span1.end_char, label='GPE')
    assert span1.start_char == span2.start_char
    assert span1.end_char == span2.end_char
    assert span2.label_ == 'GPE'
    span2 = doc.char_span(span1.start_char, span1.end_char, label='GPE', alignment_mode='strict')
    assert span1.start_char == span2.start_char
    assert span1.end_char == span2.end_char
    assert span2.label_ == 'GPE'
    span2 = doc.char_span(span1.start_char - 3, span1.end_char, label='GPE', alignment_mode='contract')
    assert span1.start_char == span2.start_char
    assert span1.end_char == span2.end_char
    assert span2.label_ == 'GPE'
    span2 = doc.char_span(span1.start_char + 1, span1.end_char, label='GPE', alignment_mode='expand')
    assert span1.start_char == span2.start_char
    assert span1.end_char == span2.end_char
    assert span2.label_ == 'GPE'
    with pytest.raises(ValueError):
        span2 = doc.char_span(span1.start_char + 1, span1.end_char, label='GPE', alignment_mode='unk')
    span2 = doc[0:2].char_span(span1.start_char - 3, span1.end_char, label='GPE', alignment_mode='contract')
    assert span1.start_char == span2.start_char
    assert span1.end_char == span2.end_char
    assert span2.label_ == 'GPE'