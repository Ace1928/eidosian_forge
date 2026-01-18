import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
def test_readonly_grid(self):
    x = np.linspace(0, 4, 5)
    y = np.linspace(0, 5, 6)
    z = np.linspace(0, 6, 7)
    points = (x, y, z)
    values = np.ones((5, 6, 7))
    point = np.array([2.21, 3.12, 1.15])
    for d in points:
        d.flags.writeable = False
    values.flags.writeable = False
    point.flags.writeable = False
    interpn(points, values, point)
    RegularGridInterpolator(points, values)(point)