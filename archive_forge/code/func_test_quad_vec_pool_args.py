import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad_vec
from multiprocessing.dummy import Pool
@pytest.mark.parametrize('extra_args', [2, (2,)])
@pytest.mark.parametrize('workers', [1, 10])
def test_quad_vec_pool_args(extra_args, workers):
    f = _func_with_args
    exact = np.array([0, 4 / 3, 8 / 3])
    res, err = quad_vec(f, 0, 1, args=extra_args, workers=workers)
    assert_allclose(res, exact, rtol=0, atol=0.0001)
    with Pool(workers) as pool:
        res, err = quad_vec(f, 0, 1, args=extra_args, workers=pool.map)
        assert_allclose(res, exact, rtol=0, atol=0.0001)