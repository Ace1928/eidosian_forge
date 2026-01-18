import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.utils._fast_dict import IntFloatDict, argmin
def test_to_arrays():
    keys_in = np.array([1, 2, 3], dtype=np.intp)
    values_in = np.array([4, 5, 6], dtype=np.float64)
    d = IntFloatDict(keys_in, values_in)
    keys_out, values_out = d.to_arrays()
    assert keys_out.dtype == keys_in.dtype
    assert values_in.dtype == values_out.dtype
    assert_array_equal(keys_out, keys_in)
    assert_allclose(values_out, values_in)