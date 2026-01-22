import numpy as np
import numpy.ma as ma
from numpy import recarray
from numpy.ma import masked, nomask
from numpy.testing import temppath
from numpy.core.records import (
from numpy.ma.mrecords import (
from numpy.ma.testutils import (
from numpy.compat import pickle
class TestMRecords:
    ilist = [1, 2, 3, 4, 5]
    flist = [1.1, 2.2, 3.3, 4.4, 5.5]
    slist = [b'one', b'two', b'three', b'four', b'five']
    ddtype = [('a', int), ('b', float), ('c', '|S8')]
    mask = [0, 1, 0, 0, 1]
    base = ma.array(list(zip(ilist, flist, slist)), mask=mask, dtype=ddtype)

    def test_byview(self):
        base = self.base
        mbase = base.view(mrecarray)
        assert_equal(mbase.recordmask, base.recordmask)
        assert_equal_records(mbase._mask, base._mask)
        assert_(isinstance(mbase._data, recarray))
        assert_equal_records(mbase._data, base._data.view(recarray))
        for field in ('a', 'b', 'c'):
            assert_equal(base[field], mbase[field])
        assert_equal_records(mbase.view(mrecarray), mbase)

    def test_get(self):
        base = self.base.copy()
        mbase = base.view(mrecarray)
        for field in ('a', 'b', 'c'):
            assert_equal(getattr(mbase, field), mbase[field])
            assert_equal(base[field], mbase[field])
        mbase_first = mbase[0]
        assert_(isinstance(mbase_first, mrecarray))
        assert_equal(mbase_first.dtype, mbase.dtype)
        assert_equal(mbase_first.tolist(), (1, 1.1, b'one'))
        assert_equal(mbase_first.recordmask, nomask)
        assert_equal(mbase_first._mask.item(), (False, False, False))
        assert_equal(mbase_first['a'], mbase['a'][0])
        mbase_last = mbase[-1]
        assert_(isinstance(mbase_last, mrecarray))
        assert_equal(mbase_last.dtype, mbase.dtype)
        assert_equal(mbase_last.tolist(), (None, None, None))
        assert_equal(mbase_last.recordmask, True)
        assert_equal(mbase_last._mask.item(), (True, True, True))
        assert_equal(mbase_last['a'], mbase['a'][-1])
        assert_(mbase_last['a'] is masked)
        mbase_sl = mbase[:2]
        assert_(isinstance(mbase_sl, mrecarray))
        assert_equal(mbase_sl.dtype, mbase.dtype)
        assert_equal(mbase_sl.recordmask, [0, 1])
        assert_equal_records(mbase_sl.mask, np.array([(False, False, False), (True, True, True)], dtype=mbase._mask.dtype))
        assert_equal_records(mbase_sl, base[:2].view(mrecarray))
        for field in ('a', 'b', 'c'):
            assert_equal(getattr(mbase_sl, field), base[:2][field])

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

    def test_set_fields_mask(self):
        base = self.base.copy()
        mbase = base.view(mrecarray)
        mbase['a'][-2] = masked
        assert_equal(mbase.a, [1, 2, 3, 4, 5])
        assert_equal(mbase.a._mask, [0, 1, 0, 1, 1])
        mbase = fromarrays([np.arange(5), np.random.rand(5)], dtype=[('a', int), ('b', float)])
        mbase['a'][-2] = masked
        assert_equal(mbase.a, [0, 1, 2, 3, 4])
        assert_equal(mbase.a._mask, [0, 0, 0, 1, 0])

    def test_set_mask(self):
        base = self.base.copy()
        mbase = base.view(mrecarray)
        mbase.mask = masked
        assert_equal(ma.getmaskarray(mbase['b']), [1] * 5)
        assert_equal(mbase['a']._mask, mbase['b']._mask)
        assert_equal(mbase['a']._mask, mbase['c']._mask)
        assert_equal(mbase._mask.tolist(), np.array([(1, 1, 1)] * 5, dtype=bool))
        mbase.mask = nomask
        assert_equal(ma.getmaskarray(mbase['c']), [0] * 5)
        assert_equal(mbase._mask.tolist(), np.array([(0, 0, 0)] * 5, dtype=bool))

    def test_set_mask_fromarray(self):
        base = self.base.copy()
        mbase = base.view(mrecarray)
        mbase.mask = [1, 0, 0, 0, 1]
        assert_equal(mbase.a.mask, [1, 0, 0, 0, 1])
        assert_equal(mbase.b.mask, [1, 0, 0, 0, 1])
        assert_equal(mbase.c.mask, [1, 0, 0, 0, 1])
        mbase.mask = [0, 0, 0, 0, 1]
        assert_equal(mbase.a.mask, [0, 0, 0, 0, 1])
        assert_equal(mbase.b.mask, [0, 0, 0, 0, 1])
        assert_equal(mbase.c.mask, [0, 0, 0, 0, 1])

    def test_set_mask_fromfields(self):
        mbase = self.base.copy().view(mrecarray)
        nmask = np.array([(0, 1, 0), (0, 1, 0), (1, 0, 1), (1, 0, 1), (0, 0, 0)], dtype=[('a', bool), ('b', bool), ('c', bool)])
        mbase.mask = nmask
        assert_equal(mbase.a.mask, [0, 0, 1, 1, 0])
        assert_equal(mbase.b.mask, [1, 1, 0, 0, 0])
        assert_equal(mbase.c.mask, [0, 0, 1, 1, 0])
        mbase.mask = False
        mbase.fieldmask = nmask
        assert_equal(mbase.a.mask, [0, 0, 1, 1, 0])
        assert_equal(mbase.b.mask, [1, 1, 0, 0, 0])
        assert_equal(mbase.c.mask, [0, 0, 1, 1, 0])

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

    def test_setslices_hardmask(self):
        base = self.base.copy()
        mbase = base.view(mrecarray)
        mbase.harden_mask()
        try:
            mbase[-2:] = (5, 5, 5)
            assert_equal(mbase.a._data, [1, 2, 3, 5, 5])
            assert_equal(mbase.b._data, [1.1, 2.2, 3.3, 5, 5.5])
            assert_equal(mbase.c._data, [b'one', b'two', b'three', b'5', b'five'])
            assert_equal(mbase.a._mask, [0, 1, 0, 0, 1])
            assert_equal(mbase.b._mask, mbase.a._mask)
            assert_equal(mbase.b._mask, mbase.c._mask)
        except NotImplementedError:
            pass
        except AssertionError:
            raise
        else:
            raise Exception('Flexible hard masks should be supported !')
        try:
            mbase[-2:] = 3
        except (NotImplementedError, TypeError):
            pass
        else:
            raise TypeError('Should have expected a readable buffer object!')

    def test_hardmask(self):
        base = self.base.copy()
        mbase = base.view(mrecarray)
        mbase.harden_mask()
        assert_(mbase._hardmask)
        mbase.mask = nomask
        assert_equal_records(mbase._mask, base._mask)
        mbase.soften_mask()
        assert_(not mbase._hardmask)
        mbase.mask = nomask
        assert_equal_records(mbase._mask, ma.make_mask_none(base.shape, base.dtype))
        assert_(ma.make_mask(mbase['b']._mask) is nomask)
        assert_equal(mbase['a']._mask, mbase['b']._mask)

    def test_pickling(self):
        base = self.base.copy()
        mrec = base.view(mrecarray)
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            _ = pickle.dumps(mrec, protocol=proto)
            mrec_ = pickle.loads(_)
            assert_equal(mrec_.dtype, mrec.dtype)
            assert_equal_records(mrec_._data, mrec._data)
            assert_equal(mrec_._mask, mrec._mask)
            assert_equal_records(mrec_._mask, mrec._mask)

    def test_filled(self):
        _a = ma.array([1, 2, 3], mask=[0, 0, 1], dtype=int)
        _b = ma.array([1.1, 2.2, 3.3], mask=[0, 0, 1], dtype=float)
        _c = ma.array(['one', 'two', 'three'], mask=[0, 0, 1], dtype='|S8')
        ddtype = [('a', int), ('b', float), ('c', '|S8')]
        mrec = fromarrays([_a, _b, _c], dtype=ddtype, fill_value=(99999, 99999.0, 'N/A'))
        mrecfilled = mrec.filled()
        assert_equal(mrecfilled['a'], np.array((1, 2, 99999), dtype=int))
        assert_equal(mrecfilled['b'], np.array((1.1, 2.2, 99999.0), dtype=float))
        assert_equal(mrecfilled['c'], np.array(('one', 'two', 'N/A'), dtype='|S8'))

    def test_tolist(self):
        _a = ma.array([1, 2, 3], mask=[0, 0, 1], dtype=int)
        _b = ma.array([1.1, 2.2, 3.3], mask=[0, 0, 1], dtype=float)
        _c = ma.array(['one', 'two', 'three'], mask=[1, 0, 0], dtype='|S8')
        ddtype = [('a', int), ('b', float), ('c', '|S8')]
        mrec = fromarrays([_a, _b, _c], dtype=ddtype, fill_value=(99999, 99999.0, 'N/A'))
        assert_equal(mrec.tolist(), [(1, 1.1, None), (2, 2.2, b'two'), (None, None, b'three')])

    def test_withnames(self):
        x = mrecarray(1, formats=float, names='base')
        x[0]['base'] = 10
        assert_equal(x['base'][0], 10)

    def test_exotic_formats(self):
        easy = mrecarray(1, dtype=[('i', int), ('s', '|S8'), ('f', float)])
        easy[0] = masked
        assert_equal(easy.filled(1).item(), (1, b'1', 1.0))
        solo = mrecarray(1, dtype=[('f0', '<f8', (2, 2))])
        solo[0] = masked
        assert_equal(solo.filled(1).item(), np.array((1,), dtype=solo.dtype).item())
        mult = mrecarray(2, dtype='i4, (2,3)float, float')
        mult[0] = masked
        mult[1] = (1, 1, 1)
        mult.filled(0)
        assert_equal_records(mult.filled(0), np.array([(0, 0, 0), (1, 1, 1)], dtype=mult.dtype))