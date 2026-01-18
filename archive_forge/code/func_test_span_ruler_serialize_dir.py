import pytest
from thinc.api import NumpyOps, get_current_ops
import spacy
from spacy import registry
from spacy.errors import MatchPatternError
from spacy.tests.util import make_tempdir
from spacy.tokens import Span
from spacy.training import Example
def test_span_ruler_serialize_dir(patterns):
    nlp = spacy.blank('xx')
    ruler = nlp.add_pipe('span_ruler')
    ruler.add_patterns(patterns)
    with make_tempdir() as d:
        ruler.to_disk(d / 'test_ruler')
        ruler.from_disk(d / 'test_ruler')
        with pytest.raises(ValueError):
            ruler.from_disk(d / 'non_existing_dir')