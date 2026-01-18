import pytest
from thinc.api import NumpyOps, get_current_ops
import spacy
from spacy import registry
from spacy.errors import MatchPatternError
from spacy.tests.util import make_tempdir
from spacy.tokens import Span
from spacy.training import Example
def test_span_ruler_properties(patterns):
    nlp = spacy.blank('xx')
    ruler = nlp.add_pipe('span_ruler', config={'overwrite': True})
    ruler.add_patterns(patterns)
    assert sorted(ruler.labels) == sorted(set([p['label'] for p in patterns]))