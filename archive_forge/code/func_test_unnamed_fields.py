import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_unnamed_fields(self):
    _, x, y, _ = self.data
    test = stack_arrays((x, x), usemask=False)
    control = np.array([1, 2, 1, 2])
    assert_equal(test, control)
    test = stack_arrays((x, y), usemask=False)
    control = np.array([1, 2, 10, 20, 30])
    assert_equal(test, control)
    test = stack_arrays((y, x), usemask=False)
    control = np.array([10, 20, 30, 1, 2])
    assert_equal(test, control)