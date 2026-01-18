import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_shapes_scalarvalue(self):
    P = BarycentricInterpolator(self.xs, self.ys)
    assert_array_equal(np.shape(P(0)), ())
    assert_array_equal(np.shape(P(np.array(0))), ())
    assert_array_equal(np.shape(P([0])), (1,))
    assert_array_equal(np.shape(P([0, 1])), (2,))