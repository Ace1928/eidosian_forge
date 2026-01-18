import os
import pytest
import sys
import numpy as np
from numpy.testing import assert_allclose
from pytest import raises as assert_raises
from scipy.sparse.linalg._svdp import _svdp
from scipy.sparse import csr_matrix, csc_matrix
@pytest.mark.parametrize('ctor', (np.array, csr_matrix, csc_matrix))
@pytest.mark.parametrize('dtype', _dtypes)
@pytest.mark.parametrize('irl', (True, False))
@pytest.mark.parametrize('which', ('LM', 'SM'))
def test_svdp(ctor, dtype, irl, which):
    np.random.seed(0)
    n, m, k = (10, 20, 3)
    if which == 'SM' and (not irl):
        message = "`which`='SM' requires irl_mode=True"
        with assert_raises(ValueError, match=message):
            check_svdp(n, m, ctor, dtype, k, irl, which)
    elif is_32bit() and is_complex_type(dtype):
        message = 'PROPACK complex-valued SVD methods not available '
        with assert_raises(TypeError, match=message):
            check_svdp(n, m, ctor, dtype, k, irl, which)
    else:
        check_svdp(n, m, ctor, dtype, k, irl, which)