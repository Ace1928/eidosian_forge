import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_solo(self):
    _, x, _, _ = self.data
    test = stack_arrays((x,))
    assert_equal(test, x)
    assert_(test is x)
    test = stack_arrays(x)
    assert_equal(test, x)
    assert_(test is x)