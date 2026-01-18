import pytest
from thinc.api import NumpyOps, get_current_ops
import spacy
from spacy import registry
from spacy.errors import MatchPatternError
from spacy.tests.util import make_tempdir
from spacy.tokens import Span
from spacy.training import Example
def test_span_ruler_existing_overwrite(patterns):
    nlp = spacy.blank('xx')
    ruler = nlp.add_pipe('span_ruler', config={'overwrite': True})
    ruler.add_patterns(patterns)
    doc = nlp.make_doc('OH HELLO WORLD bye bye')
    doc.spans['ruler'] = [doc[0:2]]
    doc = nlp(doc)
    assert len(doc.spans['ruler']) == 2
    assert doc.spans['ruler'][0].label_ == 'HELLO'
    assert doc.spans['ruler'][0].text == 'HELLO'
    assert doc.spans['ruler'][1].label_ == 'BYE'