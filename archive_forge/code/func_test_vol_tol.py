from numpy.testing import (assert_allclose,
import pytest
import numpy as np
from scipy.optimize import direct, Bounds
@pytest.mark.parametrize('vol_tol', [1e-06, 1e-08])
@pytest.mark.parametrize('locally_biased', [True, False])
def test_vol_tol(self, vol_tol, locally_biased):
    bounds = 4 * [(-10.0, 10.0)]
    res = direct(self.sphere, bounds=bounds, vol_tol=vol_tol, len_tol=0.0, locally_biased=locally_biased)
    assert res.status == 4
    assert res.success
    assert_allclose(res.x, np.zeros((4,)))
    message = f'The volume of the hyperrectangle containing the lowest function value found is below vol_tol={vol_tol}'
    assert res.message == message