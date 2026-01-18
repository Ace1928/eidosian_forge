import pytest
from thinc.api import NumpyOps, get_current_ops
import spacy
from spacy import registry
from spacy.errors import MatchPatternError
from spacy.tests.util import make_tempdir
from spacy.tokens import Span
from spacy.training import Example
def test_span_ruler_ents_overwrite_filter(overlapping_patterns):
    nlp = spacy.blank('xx')
    ruler = nlp.add_pipe('span_ruler', config={'annotate_ents': True, 'overwrite': False, 'ents_filter': {'@misc': 'spacy.prioritize_new_ents_filter.v1'}})
    ruler.add_patterns(overlapping_patterns)
    doc = nlp.make_doc('foo bar baz a b c')
    doc.ents = [Span(doc, 1, 3, label='BARBAZ'), Span(doc, 3, 6, label='ABC')]
    doc = ruler(doc)
    assert len(doc.ents) == 2
    assert doc.ents[0].label_ == 'FOOBAR'
    assert doc.ents[1].label_ == 'ABC'