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
def test_spans_override_sentiment(en_tokenizer):
    """Test span.sentiment property's default averaging behaviour"""
    text = 'good stuff bad stuff'
    tokens = en_tokenizer(text)
    tokens.vocab[tokens[0].text].sentiment = 3.0
    tokens.vocab[tokens[2].text].sentiment = -2.0
    doc = Doc(tokens.vocab, words=[t.text for t in tokens])
    doc.user_span_hooks['sentiment'] = lambda span: 10.0
    assert doc[:2].sentiment == 10.0
    assert doc[-2:].sentiment == 10.0
    assert doc[:-1].sentiment == 10.0