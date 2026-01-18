import os
import numpy as np
import tempfile
from pytest import raises as assert_raises
from numpy.testing import assert_equal, assert_
from scipy.sparse import (sparray, csc_matrix, csr_matrix, bsr_matrix, dia_matrix,
def test_save_and_load_empty():
    dense_matrix = np.zeros((4, 6))
    _check_save_and_load(dense_matrix)