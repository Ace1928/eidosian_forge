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
def test_vectors_most_similar(most_similar_vectors_data, most_similar_vectors_keys):
    v = Vectors(data=most_similar_vectors_data, keys=most_similar_vectors_keys)
    _, best_rows, _ = v.most_similar(v.data, batch_size=2, n=2, sort=True)
    assert all((row[0] == i for i, row in enumerate(best_rows)))
    with pytest.raises(ValueError):
        v.most_similar(v.data, batch_size=2, n=10, sort=True)