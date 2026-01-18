import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
@parametrize_square_sparrays
def test_spsolve(B):
    if B.__class__.__name__[:3] not in ('csc', 'csr'):
        return
    npt.assert_allclose(spla.spsolve(B, [1, 2]), np.linalg.solve(B.todense(), [1, 2]))