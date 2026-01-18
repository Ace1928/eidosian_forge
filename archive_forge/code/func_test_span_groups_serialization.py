import pytest
from spacy.tokens import Span, SpanGroup
from spacy.tokens._dict_proxies import SpanGroups
def test_span_groups_serialization(en_tokenizer):
    doc = en_tokenizer('0 1 2 3 4 5 6')
    span_groups = SpanGroups(doc)
    spans = [doc[0:2], doc[1:3]]
    sg1 = SpanGroup(doc, spans=spans)
    span_groups['key1'] = sg1
    span_groups['key2'] = sg1
    span_groups['key3'] = []
    reloaded_span_groups = SpanGroups(doc).from_bytes(span_groups.to_bytes())
    assert span_groups.keys() == reloaded_span_groups.keys()
    for key, value in span_groups.items():
        assert all((span == reloaded_span for span, reloaded_span in zip(span_groups[key], reloaded_span_groups[key])))