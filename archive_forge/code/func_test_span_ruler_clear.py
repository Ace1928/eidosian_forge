import pytest
from thinc.api import NumpyOps, get_current_ops
import spacy
from spacy import registry
from spacy.errors import MatchPatternError
from spacy.tests.util import make_tempdir
from spacy.tokens import Span
from spacy.training import Example
def test_span_ruler_clear(patterns):
    nlp = spacy.blank('xx')
    ruler = nlp.add_pipe('span_ruler')
    ruler.add_patterns(patterns)
    assert len(ruler.labels) == 4
    doc = nlp('hello world')
    assert len(doc.spans['ruler']) == 1
    ruler.clear()
    assert len(ruler.labels) == 0
    with pytest.warns(UserWarning):
        doc = nlp('hello world')
    assert len(doc.spans['ruler']) == 0