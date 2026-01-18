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
def test_vocab_add_vector():
    vocab = Vocab(vectors_name='test_vocab_add_vector')
    data = OPS.xp.ndarray((5, 3), dtype='f')
    data[0] = 1.0
    data[1] = 2.0
    vocab.set_vector('cat', data[0])
    vocab.set_vector('dog', data[1])
    cat = vocab['cat']
    assert list(cat.vector) == [1.0, 1.0, 1.0]
    dog = vocab['dog']
    assert list(dog.vector) == [2.0, 2.0, 2.0]
    with pytest.raises(ValueError):
        vocab.vectors.add(vocab['hamster'].orth, row=1000000)