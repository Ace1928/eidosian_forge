import math
import numpy as np
from numpy import array, eye, exp, random
from numpy.testing import (
from scipy.sparse import csc_matrix, csc_array, SparseEfficiencyWarning
from scipy.sparse._construct import eye as speye
from scipy.sparse.linalg._matfuncs import (expm, _expm,
from scipy.sparse._sputils import matrix
from scipy.linalg import logm
from scipy.special import factorial, binom
import scipy.sparse
import scipy.sparse.linalg
def test_padecases_dtype_complex(self):
    for dtype in [np.complex64, np.complex128]:
        for scale in [0.01, 0.1, 0.5, 1, 10]:
            A = scale * eye(3, dtype=dtype)
            observed = expm(A)
            expected = exp(scale, dtype=dtype) * eye(3, dtype=dtype)
            assert_array_almost_equal_nulp(observed, expected, nulp=100)