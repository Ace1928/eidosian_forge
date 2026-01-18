import pytest
from thinc.api import NumpyOps, get_current_ops
import spacy
from spacy import registry
from spacy.errors import MatchPatternError
from spacy.tests.util import make_tempdir
from spacy.tokens import Span
from spacy.training import Example
def test_span_ruler_add_empty(patterns):
    """Test that patterns don't get added excessively."""
    nlp = spacy.blank('xx')
    ruler = nlp.add_pipe('span_ruler', config={'validate': True})
    ruler.add_patterns(patterns)
    pattern_count = sum((len(mm) for mm in ruler.matcher._patterns.values()))
    assert pattern_count > 0
    ruler.add_patterns([])
    after_count = sum((len(mm) for mm in ruler.matcher._patterns.values()))
    assert after_count == pattern_count