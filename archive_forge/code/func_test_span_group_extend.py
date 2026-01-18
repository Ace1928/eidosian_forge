from random import Random
from typing import List
import pytest
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, SpanGroup
from spacy.util import filter_spans
def test_span_group_extend(doc):
    span_group_1 = doc.spans['SPANS'].copy()
    spans = [doc[0:5], doc[0:6]]
    span_group_2 = SpanGroup(doc, name='MORE_SPANS', attrs={'key': 'new_value', 'new_key': 'new_value'}, spans=spans)
    span_group_1_expected = span_group_1._concat(span_group_2)
    span_group_1.extend(span_group_2)
    assert len(span_group_1) == len(span_group_1_expected)
    assert span_group_1.attrs == {'key': 'value', 'new_key': 'new_value'}
    assert list(span_group_1) == list(span_group_1_expected)
    span_group_1 = doc.spans['SPANS']
    span_group_1.extend(spans)
    assert len(span_group_1) == len(span_group_1_expected)
    assert span_group_1.attrs == {'key': 'value'}
    assert list(span_group_1) == list(span_group_1_expected)