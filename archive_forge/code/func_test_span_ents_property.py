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
def test_span_ents_property(doc):
    doc.ents = [(doc.vocab.strings['PRODUCT'], 0, 1), (doc.vocab.strings['PRODUCT'], 7, 8), (doc.vocab.strings['PRODUCT'], 11, 14)]
    assert len(list(doc.ents)) == 3
    sentences = list(doc.sents)
    assert len(sentences) == 3
    assert len(sentences[0].ents) == 1
    assert sentences[0].ents[0].text == 'This'
    assert sentences[0].ents[0].label_ == 'PRODUCT'
    assert sentences[0].ents[0].start == 0
    assert sentences[0].ents[0].end == 1
    assert len(sentences[1].ents) == 1
    assert sentences[1].ents[0].text == 'another'
    assert sentences[1].ents[0].label_ == 'PRODUCT'
    assert sentences[1].ents[0].start == 7
    assert sentences[1].ents[0].end == 8
    assert sentences[2].ents[0].text == 'a third.'
    assert sentences[2].ents[0].label_ == 'PRODUCT'
    assert sentences[2].ents[0].start == 11
    assert sentences[2].ents[0].end == 14