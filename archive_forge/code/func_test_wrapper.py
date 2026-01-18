import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_wrapper(self):
    P = BarycentricInterpolator(self.xs, self.ys)
    bi = barycentric_interpolate
    assert_allclose(P(self.test_xs), bi(self.xs, self.ys, self.test_xs))
    assert_allclose(P.derivative(self.test_xs, 2), bi(self.xs, self.ys, self.test_xs, der=2))
    assert_allclose(P.derivatives(self.test_xs, 2), bi(self.xs, self.ys, self.test_xs, der=[0, 1]))