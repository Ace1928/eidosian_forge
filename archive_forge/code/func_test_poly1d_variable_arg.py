import numpy as np
from numpy.testing import (
import pytest
def test_poly1d_variable_arg(self):
    q = np.poly1d([1.0, 2, 3], variable='y')
    assert_equal(str(q), '   2\n1 y + 2 y + 3')
    q = np.poly1d([1.0, 2, 3], variable='lambda')
    assert_equal(str(q), '        2\n1 lambda + 2 lambda + 3')