import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from scipy.optimize._pava_pybind import pava
from scipy.optimize import isotonic_regression
@pytest.mark.parametrize(('y', 'w', 'msg'), [([[0, 1]], None, 'array has incorrect number of dimensions: 2; expected 1'), ([0, 1], [[1, 2]], 'Input arrays y and w must have one dimension of equal length'), ([0, 1], [1], 'Input arrays y and w must have one dimension of equal length'), (1, 2, 'Input arrays y and w must have one dimension of equal length'), ([0, 1], [0, 1], 'Weights w must be strictly positive')])
def test_raise_error(self, y, w, msg):
    with pytest.raises(ValueError, match=msg):
        isotonic_regression(y=y, weights=w)