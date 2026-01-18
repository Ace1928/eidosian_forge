import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_simple_flexible(self):
    a = np.array([(1, 10.0), (2, 20.0)], dtype=[('A', int), ('B', float)])
    b = np.zeros((3,), dtype=a.dtype)
    test = recursive_fill_fields(a, b)
    control = np.array([(1, 10.0), (2, 20.0), (0, 0.0)], dtype=[('A', int), ('B', float)])
    assert_equal(test, control)