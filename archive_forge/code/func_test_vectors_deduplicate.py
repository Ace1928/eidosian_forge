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
def test_vectors_deduplicate():
    data = OPS.asarray([[1, 1], [2, 2], [3, 4], [1, 1], [3, 4]], dtype='f')
    v = Vectors(data=data, keys=['a1', 'b1', 'c1', 'a2', 'c2'])
    vocab = Vocab()
    vocab.vectors = v
    assert vocab.vectors.key2row[v.strings['a1']] != vocab.vectors.key2row[v.strings['a2']]
    assert vocab.vectors.key2row[v.strings['c1']] != vocab.vectors.key2row[v.strings['c2']]
    vocab.deduplicate_vectors()
    assert vocab.vectors.shape[0] == 3
    assert_equal(numpy.unique(OPS.to_numpy(vocab.vectors.data), axis=0), OPS.to_numpy(vocab.vectors.data))
    assert vocab.vectors.key2row[v.strings['a1']] == vocab.vectors.key2row[v.strings['a2']]
    assert vocab.vectors.key2row[v.strings['c1']] == vocab.vectors.key2row[v.strings['c2']]
    vocab_b = vocab.to_bytes()
    vocab.deduplicate_vectors()
    assert vocab_b == vocab.to_bytes()