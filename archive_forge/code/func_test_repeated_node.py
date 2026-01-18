import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_repeated_node(self):
    xis = np.array([0.1, 0.5, 0.9, 0.5])
    ys = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError, match='Interpolation points xi must be distinct.'):
        BarycentricInterpolator(xis, ys)