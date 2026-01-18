import numpy
import pytest
from numpy.testing import assert_array_equal
from thinc.api import get_current_ops
from spacy.attrs import LENGTH, ORTH
from spacy.lang.en import English
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans
from spacy.vocab import Vocab
from ..util import add_vecs_to_vocab
from .test_underscore import clean_underscore  # noqa: F401
def test_spans_lca_matrix(en_tokenizer):
    """Test span's lca matrix generation"""
    tokens = en_tokenizer('the lazy dog slept')
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], heads=[2, 2, 3, 3], deps=['dep'] * 4)
    lca = doc[:2].get_lca_matrix()
    assert lca.shape == (2, 2)
    assert lca[0, 0] == 0
    assert lca[0, 1] == -1
    assert lca[1, 0] == -1
    assert lca[1, 1] == 1
    lca = doc[1:].get_lca_matrix()
    assert lca.shape == (3, 3)
    assert lca[0, 0] == 0
    assert lca[0, 1] == 1
    assert lca[0, 2] == 2
    lca = doc[2:].get_lca_matrix()
    assert lca.shape == (2, 2)
    assert lca[0, 0] == 0
    assert lca[0, 1] == 1
    assert lca[1, 0] == 1
    assert lca[1, 1] == 1
    tokens = en_tokenizer('I like New York in Autumn')
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], heads=[1, 1, 3, 1, 3, 4], deps=['dep'] * len(tokens))
    lca = doc[1:4].get_lca_matrix()
    assert_array_equal(lca, numpy.asarray([[0, 0, 0], [0, 1, 2], [0, 2, 2]]))