from __future__ import annotations
from typing import Callable
import pytest
from itertools import product
from numpy.testing import assert_allclose, suppress_warnings
from scipy import special
from scipy.special import cython_special
def _generate_test_points(typecodes):
    axes = tuple((TEST_POINTS[x] for x in typecodes))
    pts = list(product(*axes))
    return pts