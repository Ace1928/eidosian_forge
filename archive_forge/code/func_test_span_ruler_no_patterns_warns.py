import pytest
from thinc.api import NumpyOps, get_current_ops
import spacy
from spacy import registry
from spacy.errors import MatchPatternError
from spacy.tests.util import make_tempdir
from spacy.tokens import Span
from spacy.training import Example
def test_span_ruler_no_patterns_warns():
    nlp = spacy.blank('xx')
    ruler = nlp.add_pipe('span_ruler')
    assert len(ruler) == 0
    assert len(ruler.labels) == 0
    assert nlp.pipe_names == ['span_ruler']
    with pytest.warns(UserWarning):
        doc = nlp('hello world bye bye')
    assert len(doc.spans['ruler']) == 0