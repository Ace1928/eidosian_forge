import numpy
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
from thinc.api import NumpyOps, get_current_ops
from spacy.lang.en import English
from spacy.strings import hash_string  # type: ignore
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.training.initialize import convert_vectors
from spacy.vectors import Vectors
from spacy.vocab import Vocab
from ..util import add_vecs_to_vocab, get_cosine, make_tempdir
@pytest.mark.parametrize('text', [['apple', 'orange', 'juice']])
def test_vectors_token_span_similarity(vocab, text):
    doc = Doc(vocab, words=text)
    assert doc[0].similarity(doc[1:3]) == doc[1:3].similarity(doc[0])
    assert -1.0 < doc[0].similarity(doc[1:3]) < 1.0