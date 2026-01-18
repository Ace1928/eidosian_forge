import numpy
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from thinc.api import NumpyOps, Ragged, get_current_ops
from spacy import util
from spacy.lang.en import English
from spacy.language import Language
from spacy.tokens import SpanGroup
from spacy.tokens._dict_proxies import SpanGroups
from spacy.training import Example
from spacy.util import fix_random_seed, make_tempdir, registry
def test_preset_spans_suggester():
    nlp = Language()
    docs = [nlp('This is an example.'), nlp('This is the second example.')]
    docs[0].spans[SPAN_KEY] = [docs[0][3:4]]
    docs[1].spans[SPAN_KEY] = [docs[1][0:4], docs[1][3:5]]
    suggester = registry.misc.get('spacy.preset_spans_suggester.v1')(spans_key=SPAN_KEY)
    candidates = suggester(docs)
    assert type(candidates) == Ragged
    assert len(candidates) == 2
    assert list(candidates.dataXd[0]) == [3, 4]
    assert list(candidates.dataXd[1]) == [0, 4]
    assert list(candidates.dataXd[2]) == [3, 5]
    assert list(candidates.lengths) == [1, 2]