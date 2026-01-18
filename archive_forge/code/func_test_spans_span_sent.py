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
def test_spans_span_sent(doc, doc_not_parsed):
    """Test span.sent property"""
    assert len(list(doc.sents))
    assert doc[:2].sent.root.text == 'is'
    assert doc[:2].sent.text == 'This is a sentence.'
    assert doc[6:7].sent.root.left_edge.text == 'This'
    assert doc[0:len(doc)].sent == list(doc.sents)[0]
    assert list(doc[0:len(doc)].sents) == list(doc.sents)
    with pytest.raises(ValueError):
        doc_not_parsed[:2].sent
    doc_not_parsed[0].is_sent_start = True
    doc_not_parsed[5].is_sent_start = True
    assert doc_not_parsed[1:3].sent == doc_not_parsed[0:5]
    assert doc_not_parsed[10:14].sent == doc_not_parsed[5:]