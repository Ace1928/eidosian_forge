import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_wmasked_arrays(self):
    _, x, _, _ = self.data
    mx = ma.array([1, 2, 3], mask=[1, 0, 0])
    test = merge_arrays((x, mx), usemask=True)
    control = ma.array([(1, 1), (2, 2), (-1, 3)], mask=[(0, 1), (0, 0), (1, 0)], dtype=[('f0', int), ('f1', int)])
    assert_equal(test, control)
    test = merge_arrays((x, mx), usemask=True, asrecarray=True)
    assert_equal(test, control)
    assert_(isinstance(test, MaskedRecords))