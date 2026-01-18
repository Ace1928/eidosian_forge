import numpy as np
from numpy.testing import assert_array_almost_equal, assert_
from scipy.sparse import csr_matrix, hstack
import pytest

    Tests if hstack properly promotes to indices and indptr arrays to np.int64
    when using np.int32 during concatenation would result in either array
    overflowing.
    