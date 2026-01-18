from numpy.testing import (assert_allclose,
import pytest
import numpy as np
from scipy.optimize import direct, Bounds
@pytest.mark.parametrize('vol_tol', [-1, 2])
def test_vol_tol_validation(self, vol_tol):
    error_msg = 'vol_tol must be between 0 and 1.'
    with pytest.raises(ValueError, match=error_msg):
        direct(self.styblinski_tang, self.bounds_stylinski_tang, vol_tol=vol_tol)