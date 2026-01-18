import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
@parametrize_eig_sparrays
def test_svds(X):
    u, s, vh = spla.svds(X, k=3)
    u2, s2, vh2 = np.linalg.svd(X.todense())
    s = np.sort(s)
    s2 = np.sort(s2[:3])
    npt.assert_allclose(s, s2, atol=0.001)