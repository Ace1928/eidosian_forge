import pytest
from thinc.api import NumpyOps, get_current_ops
import spacy
from spacy import registry
from spacy.errors import MatchPatternError
from spacy.tests.util import make_tempdir
from spacy.tokens import Span
from spacy.training import Example
def test_span_ruler_remove_all_patterns(person_org_date_patterns):
    nlp = spacy.blank('xx')
    ruler = nlp.add_pipe('span_ruler')
    ruler.add_patterns(person_org_date_patterns)
    assert len(ruler.patterns) == 4
    ruler.remove('PERSON')
    assert len(ruler.patterns) == 3
    ruler.remove('ORG')
    assert len(ruler.patterns) == 1
    ruler.remove('DATE')
    assert len(ruler.patterns) == 0
    with pytest.warns(UserWarning):
        doc = ruler(nlp.make_doc('Dina founded the company ACME on June 14th'))
        assert len(doc.spans['ruler']) == 0