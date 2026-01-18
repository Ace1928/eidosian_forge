import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_w_singlefield(self):
    test = merge_arrays((np.array([1, 2]).view([('a', int)]), np.array([10.0, 20.0, 30.0])))
    control = ma.array([(1, 10.0), (2, 20.0), (-1, 30.0)], mask=[(0, 0), (0, 0), (1, 0)], dtype=[('a', int), ('f1', float)])
    assert_equal(test, control)