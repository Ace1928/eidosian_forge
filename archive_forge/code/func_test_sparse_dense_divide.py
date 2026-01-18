import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
@parametrize_sparrays
def test_sparse_dense_divide(A):
    with pytest.warns(RuntimeWarning):
        assert isinstance(A / A.todense(), scipy.sparse.sparray)