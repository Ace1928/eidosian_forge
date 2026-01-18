import pytest
from thinc.api import NumpyOps, get_current_ops
import spacy
from spacy import registry
from spacy.errors import MatchPatternError
from spacy.tests.util import make_tempdir
from spacy.tokens import Span
from spacy.training import Example
def test_span_ruler_remove_several_patterns(person_org_patterns):
    nlp = spacy.blank('xx')
    ruler = nlp.add_pipe('span_ruler')
    ruler.add_patterns(person_org_patterns)
    doc = ruler(nlp.make_doc('Dina founded the company ACME.'))
    assert len(ruler.patterns) == 3
    assert len(doc.spans['ruler']) == 2
    assert doc.spans['ruler'][0].label_ == 'PERSON'
    assert doc.spans['ruler'][0].text == 'Dina'
    assert doc.spans['ruler'][1].label_ == 'ORG'
    assert doc.spans['ruler'][1].text == 'ACME'
    ruler.remove('PERSON')
    doc = ruler(nlp.make_doc('Dina founded the company ACME'))
    assert len(ruler.patterns) == 2
    assert len(doc.spans['ruler']) == 1
    assert doc.spans['ruler'][0].label_ == 'ORG'
    assert doc.spans['ruler'][0].text == 'ACME'
    ruler.remove('ORG')
    with pytest.warns(UserWarning):
        doc = ruler(nlp.make_doc('Dina founded the company ACME'))
        assert len(ruler.patterns) == 0
        assert len(doc.spans['ruler']) == 0