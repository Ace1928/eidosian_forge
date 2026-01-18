import pytest
from thinc.api import NumpyOps, get_current_ops
import spacy
from spacy import registry
from spacy.errors import MatchPatternError
from spacy.tests.util import make_tempdir
from spacy.tokens import Span
from spacy.training import Example
def test_span_ruler_validate():
    nlp = spacy.blank('xx')
    ruler = nlp.add_pipe('span_ruler')
    validated_ruler = nlp.add_pipe('span_ruler', name='validated_span_ruler', config={'validate': True})
    valid_pattern = {'label': 'HELLO', 'pattern': [{'LOWER': 'HELLO'}]}
    invalid_pattern = {'label': 'HELLO', 'pattern': [{'ASDF': 'HELLO'}]}
    with pytest.raises(ValueError):
        ruler.add_patterns([invalid_pattern])
    validated_ruler.add_patterns([valid_pattern])
    with pytest.raises(MatchPatternError):
        validated_ruler.add_patterns([invalid_pattern])