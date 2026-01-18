import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_shapes_vectorvalue_derivative(self):
    P = BarycentricInterpolator(self.xs, np.outer(self.ys, np.arange(3)))
    n = P.n
    assert_array_equal(np.shape(P.derivatives(0)), (n, 3))
    assert_array_equal(np.shape(P.derivatives([0])), (n, 1, 3))
    assert_array_equal(np.shape(P.derivatives([0, 1])), (n, 2, 3))