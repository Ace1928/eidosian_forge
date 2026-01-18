import numpy as np
from numpy.testing import assert_equal, assert_array_equal
import pytest
from scipy.stats import rankdata, tiecorrect
from scipy._lib._util import np_long
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('method, dtype', zip(methods, dtypes))
def test_size_0_axis(self, axis, method, dtype):
    shape = (3, 0)
    data = np.zeros(shape)
    r = rankdata(data, method=method, axis=axis)
    assert_equal(r.shape, shape)
    assert_equal(r.dtype, dtype)