from random import Random
from typing import List
import pytest
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, SpanGroup
from spacy.util import filter_spans
def test_span_group_has_overlap(doc):
    span_group = doc.spans['SPANS']
    assert span_group.has_overlap