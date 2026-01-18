import pytest
from thinc.api import NumpyOps, get_current_ops
import spacy
from spacy import registry
from spacy.errors import MatchPatternError
from spacy.tests.util import make_tempdir
from spacy.tokens import Span
from spacy.training import Example
def test_span_ruler_spans_filter(overlapping_patterns):
    nlp = spacy.blank('xx')
    ruler = nlp.add_pipe('span_ruler', config={'spans_filter': {'@misc': 'spacy.first_longest_spans_filter.v1'}})
    ruler.add_patterns(overlapping_patterns)
    doc = ruler(nlp.make_doc('foo bar baz'))
    assert len(doc.spans['ruler']) == 1
    assert doc.spans['ruler'][0].label_ == 'FOOBAR'