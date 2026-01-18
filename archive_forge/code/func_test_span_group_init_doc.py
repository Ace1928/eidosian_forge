from random import Random
from typing import List
import pytest
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, SpanGroup
from spacy.util import filter_spans
def test_span_group_init_doc(en_tokenizer):
    """Test that all spans must come from the specified doc."""
    doc1 = en_tokenizer('a b c')
    doc2 = en_tokenizer('a b c')
    span_group = SpanGroup(doc1, spans=[doc1[0:1], doc1[1:2]])
    with pytest.raises(ValueError):
        span_group = SpanGroup(doc1, spans=[doc1[0:1], doc2[1:2]])