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
def test_sent(en_tokenizer):
    doc = en_tokenizer('Check span.sent raises error if doc is not sentencized.')
    span = doc[1:3]
    assert not span.doc.has_annotation('SENT_START')
    with pytest.raises(ValueError):
        span.sent