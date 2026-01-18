import os
import numpy as np
import tempfile
from pytest import raises as assert_raises
from numpy.testing import assert_equal, assert_
from scipy.sparse import (sparray, csc_matrix, csr_matrix, bsr_matrix, dia_matrix,
def test_py23_compatibility():
    a = load_npz(os.path.join(DATA_DIR, 'csc_py2.npz'))
    b = load_npz(os.path.join(DATA_DIR, 'csc_py3.npz'))
    c = csc_matrix([[0]])
    assert_equal(a.toarray(), c.toarray())
    assert_equal(b.toarray(), c.toarray())