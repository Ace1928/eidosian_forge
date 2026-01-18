import pytest
from thinc.api import NumpyOps, get_current_ops
import spacy
from spacy import registry
from spacy.errors import MatchPatternError
from spacy.tests.util import make_tempdir
from spacy.tokens import Span
from spacy.training import Example
@pytest.mark.parametrize('n_process', [1, 2])
def test_span_ruler_multiprocessing(n_process):
    if isinstance(get_current_ops, NumpyOps) or n_process < 2:
        texts = ['I enjoy eating Pizza Hut pizza.']
        patterns = [{'label': 'FASTFOOD', 'pattern': 'Pizza Hut'}]
        nlp = spacy.blank('xx')
        ruler = nlp.add_pipe('span_ruler')
        ruler.add_patterns(patterns)
        for doc in nlp.pipe(texts, n_process=2):
            for ent in doc.spans['ruler']:
                assert ent.label_ == 'FASTFOOD'