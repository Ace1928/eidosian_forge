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
@pytest.mark.parametrize('text', [['apple', 'and', 'orange']])
def test_vectors_span_vector(vocab, text):
    span = Doc(vocab, words=text)[0:2]
    assert list(span.vector)
    assert span.vector_norm