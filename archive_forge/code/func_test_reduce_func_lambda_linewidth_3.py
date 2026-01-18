import numpy as np
from ..._shared.testing import assert_equal, assert_almost_equal
from ..profile import profile_line
def test_reduce_func_lambda_linewidth_3():

    def reduce_func(x):
        return x + x ** 2
    prof = profile_line(pyth_image, (1, 2), (4, 2), linewidth=3, order=0, reduce_func=reduce_func, mode='constant')
    expected_prof = np.apply_along_axis(reduce_func, arr=pyth_image[1:5, 1:4], axis=1)
    assert_almost_equal(prof, expected_prof)