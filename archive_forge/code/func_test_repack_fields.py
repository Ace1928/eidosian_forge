import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_repack_fields(self):
    dt = np.dtype('u1,f4,i8', align=True)
    a = np.zeros(2, dtype=dt)
    assert_equal(repack_fields(dt), np.dtype('u1,f4,i8'))
    assert_equal(repack_fields(a).itemsize, 13)
    assert_equal(repack_fields(repack_fields(dt), align=True), dt)
    dt = np.dtype((np.record, dt))
    assert_(repack_fields(dt).type is np.record)