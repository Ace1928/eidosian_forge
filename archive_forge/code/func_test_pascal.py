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
def test_pascal(self):
    for scale in [1.0, 0.001, 1e-06]:
        for n in range(0, 80, 3):
            sc = scale ** np.arange(n, -1, -1)
            if np.any(sc < 1e-300):
                break
            A = np.diag(np.arange(1, n + 1), -1) * scale
            B = expm(A)
            got = B
            expected = binom(np.arange(n + 1)[:, None], np.arange(n + 1)[None, :]) * sc[None, :] / sc[:, None]
            atol = 1e-13 * abs(expected).max()
            assert_allclose(got, expected, atol=atol)