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
def test_vectors_serialize():
    data = OPS.asarray([[4, 2, 2, 2], [4, 2, 2, 2], [1, 1, 1, 1]], dtype='f')
    v = Vectors(data=data, keys=['A', 'B', 'C'])
    b = v.to_bytes()
    v_r = Vectors()
    v_r.from_bytes(b)
    assert_equal(OPS.to_numpy(v.data), OPS.to_numpy(v_r.data))
    assert v.key2row == v_r.key2row
    v.resize((5, 4))
    v_r.resize((5, 4))
    row = v.add('D', vector=OPS.asarray([1, 2, 3, 4], dtype='f'))
    row_r = v_r.add('D', vector=OPS.asarray([1, 2, 3, 4], dtype='f'))
    assert row == row_r
    assert_equal(OPS.to_numpy(v.data), OPS.to_numpy(v_r.data))
    assert v.is_full == v_r.is_full
    with make_tempdir() as d:
        v.to_disk(d)
        v_r.from_disk(d)
        assert_equal(OPS.to_numpy(v.data), OPS.to_numpy(v_r.data))
        assert v.key2row == v_r.key2row
        v.resize((5, 4))
        v_r.resize((5, 4))
        row = v.add('D', vector=OPS.asarray([10, 20, 30, 40], dtype='f'))
        row_r = v_r.add('D', vector=OPS.asarray([10, 20, 30, 40], dtype='f'))
        assert row == row_r
        assert_equal(OPS.to_numpy(v.data), OPS.to_numpy(v_r.data))
        assert v.attr == v_r.attr