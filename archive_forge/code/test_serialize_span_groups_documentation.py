import pytest
from spacy.tokens import Span, SpanGroup
from spacy.tokens._dict_proxies import SpanGroups
Test backwards-compatibility of `SpanGroups` deserialization.
    This uses serializations (bytes) from a prior version of spaCy (before 3.3.1).

    spans_bytes (bytes): Serialized `SpanGroups` object.
    doc_text (str): Doc text.
    expected_spangroups (dict):
        Dict mapping every expected (after deserialization) `SpanGroups` key
        to a SpanGroup's "args", where a SpanGroup's args are given as a dict:
          {"name": span_group.name,
           "spans": [(span0.start, span0.end), ...]}
    expected_warning (bool): Whether a warning is to be expected from .from_bytes()
        --i.e. if more than 1 SpanGroup has the same .name within the `SpanGroups`.
    