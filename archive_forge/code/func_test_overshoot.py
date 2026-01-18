import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_overshoot(self):
    p, xi, yi = self._make_random()
    for i in range(len(xi) - 1):
        x1, x2 = (xi[i], xi[i + 1])
        y1, y2 = (yi[i], yi[i + 1])
        if y1 > y2:
            y1, y2 = (y2, y1)
        xp = np.linspace(x1, x2, 10)
        yp = p(xp)
        assert_(((y1 <= yp + 1e-15) & (yp <= y2 + 1e-15)).all())