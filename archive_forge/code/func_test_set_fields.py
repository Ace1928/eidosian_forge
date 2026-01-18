import numpy as np
import numpy.ma as ma
from numpy import recarray
from numpy.ma import masked, nomask
from numpy.testing import temppath
from numpy.core.records import (
from numpy.ma.mrecords import (
from numpy.ma.testutils import (
from numpy.compat import pickle
def test_set_fields(self):
    base = self.base.copy()
    mbase = base.view(mrecarray)
    mbase = mbase.copy()
    mbase.fill_value = (999999, 1e+20, 'N/A')
    mbase.a._data[:] = 5
    assert_equal(mbase['a']._data, [5, 5, 5, 5, 5])
    assert_equal(mbase['a']._mask, [0, 1, 0, 0, 1])
    mbase.a = 1
    assert_equal(mbase['a']._data, [1] * 5)
    assert_equal(ma.getmaskarray(mbase['a']), [0] * 5)
    assert_equal(mbase.recordmask, [False] * 5)
    assert_equal(mbase._mask.tolist(), np.array([(0, 0, 0), (0, 1, 1), (0, 0, 0), (0, 0, 0), (0, 1, 1)], dtype=bool))
    mbase.c = masked
    assert_equal(mbase.c.mask, [1] * 5)
    assert_equal(mbase.c.recordmask, [1] * 5)
    assert_equal(ma.getmaskarray(mbase['c']), [1] * 5)
    assert_equal(ma.getdata(mbase['c']), [b'N/A'] * 5)
    assert_equal(mbase._mask.tolist(), np.array([(0, 0, 1), (0, 1, 1), (0, 0, 1), (0, 0, 1), (0, 1, 1)], dtype=bool))
    mbase = base.view(mrecarray).copy()
    mbase.a[3:] = 5
    assert_equal(mbase.a, [1, 2, 3, 5, 5])
    assert_equal(mbase.a._mask, [0, 1, 0, 0, 0])
    mbase.b[3:] = masked
    assert_equal(mbase.b, base['b'])
    assert_equal(mbase.b._mask, [0, 1, 0, 1, 1])
    ndtype = [('alpha', '|S1'), ('num', int)]
    data = ma.array([('a', 1), ('b', 2), ('c', 3)], dtype=ndtype)
    rdata = data.view(MaskedRecords)
    val = ma.array([10, 20, 30], mask=[1, 0, 0])
    rdata['num'] = val
    assert_equal(rdata.num, val)
    assert_equal(rdata.num.mask, [1, 0, 0])