import pytest
from thinc.api import NumpyOps, get_current_ops
import spacy
from spacy import registry
from spacy.errors import MatchPatternError
from spacy.tests.util import make_tempdir
from spacy.tokens import Span
from spacy.training import Example
def test_span_ruler_init(patterns):
    nlp = spacy.blank('xx')
    ruler = nlp.add_pipe('span_ruler')
    ruler.add_patterns(patterns)
    assert len(ruler) == len(patterns)
    assert len(ruler.labels) == 4
    assert 'HELLO' in ruler
    assert 'BYE' in ruler
    doc = nlp('hello world bye bye')
    assert len(doc.spans['ruler']) == 2
    assert doc.spans['ruler'][0].label_ == 'HELLO'
    assert doc.spans['ruler'][0].id_ == 'hello1'
    assert doc.spans['ruler'][1].label_ == 'BYE'
    assert doc.spans['ruler'][1].id_ == ''