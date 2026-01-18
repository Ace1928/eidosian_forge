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
def test_span_group_copy(doc):
    doc.spans['test'] = [doc[0:1], doc[2:4]]
    assert len(doc.spans['test']) == 2
    doc_copy = doc.copy()
    assert len(doc_copy.spans['test']) == 2
    doc.spans['test'].append(doc[3:4])
    assert len(doc.spans['test']) == 3
    assert len(doc_copy.spans['test']) == 2