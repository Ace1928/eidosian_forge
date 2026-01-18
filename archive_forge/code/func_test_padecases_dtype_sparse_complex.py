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
def test_padecases_dtype_sparse_complex(self):
    dtype = np.complex128
    for scale in [0.01, 0.1, 0.5, 1, 10]:
        a = scale * speye(3, 3, dtype=dtype, format='csc')
        e = exp(scale) * eye(3, dtype=dtype)
        with suppress_warnings() as sup:
            sup.filter(SparseEfficiencyWarning, 'Changing the sparsity structure of a csc_matrix is expensive.')
            assert_array_almost_equal_nulp(expm(a).toarray(), e, nulp=100)