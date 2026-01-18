import numpy as np
import numpy.ma as ma
from numpy import recarray
from numpy.ma import masked, nomask
from numpy.testing import temppath
from numpy.core.records import (
from numpy.ma.mrecords import (
from numpy.ma.testutils import (
from numpy.compat import pickle
def test_set_elements(self):
    base = self.base.copy()
    mbase = base.view(mrecarray).copy()
    mbase[-2] = masked
    assert_equal(mbase._mask.tolist(), np.array([(0, 0, 0), (1, 1, 1), (0, 0, 0), (1, 1, 1), (1, 1, 1)], dtype=bool))
    assert_equal(mbase.recordmask, [0, 1, 0, 1, 1])
    mbase = base.view(mrecarray).copy()
    mbase[:2] = (5, 5, 5)
    assert_equal(mbase.a._data, [5, 5, 3, 4, 5])
    assert_equal(mbase.a._mask, [0, 0, 0, 0, 1])
    assert_equal(mbase.b._data, [5.0, 5.0, 3.3, 4.4, 5.5])
    assert_equal(mbase.b._mask, [0, 0, 0, 0, 1])
    assert_equal(mbase.c._data, [b'5', b'5', b'three', b'four', b'five'])
    assert_equal(mbase.b._mask, [0, 0, 0, 0, 1])
    mbase = base.view(mrecarray).copy()
    mbase[:2] = masked
    assert_equal(mbase.a._data, [1, 2, 3, 4, 5])
    assert_equal(mbase.a._mask, [1, 1, 0, 0, 1])
    assert_equal(mbase.b._data, [1.1, 2.2, 3.3, 4.4, 5.5])
    assert_equal(mbase.b._mask, [1, 1, 0, 0, 1])
    assert_equal(mbase.c._data, [b'one', b'two', b'three', b'four', b'five'])
    assert_equal(mbase.b._mask, [1, 1, 0, 0, 1])