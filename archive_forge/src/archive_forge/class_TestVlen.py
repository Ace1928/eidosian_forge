from itertools import count
import platform
import numpy as np
import h5py
from .common import ut, TestCase
class TestVlen(TestCase):
    """
        Check that storage of vlen strings is carried out correctly.
    """

    def assertVlenArrayEqual(self, dset, arr, message=None, precision=None):
        assert dset.shape == arr.shape, 'Shape mismatch (%s vs %s)%s' % (dset.shape, arr.shape, message)
        for i, d, a in zip(count(), dset, arr):
            self.assertArrayEqual(d, a, message, precision)

    def test_compound(self):
        fields = []
        fields.append(('field_1', h5py.string_dtype()))
        fields.append(('field_2', np.int32))
        dt = np.dtype(fields)
        self.f['mytype'] = np.dtype(dt)
        dt_out = self.f['mytype'].dtype.fields['field_1'][0]
        string_inf = h5py.check_string_dtype(dt_out)
        self.assertEqual(string_inf.encoding, 'utf-8')

    def test_compound_vlen_bool(self):
        vidt = h5py.vlen_dtype(np.uint8)

        def a(items):
            return np.array(items, dtype=np.uint8)
        f = self.f
        dt_vb = np.dtype([('foo', vidt), ('logical', bool)])
        vb = f.create_dataset('dt_vb', shape=(4,), dtype=dt_vb)
        data = np.array([(a([1, 2, 3]), True), (a([1]), False), (a([1, 5]), True), (a([]), False)], dtype=dt_vb)
        vb[:] = data
        actual = f['dt_vb'][:]
        self.assertVlenArrayEqual(data['foo'], actual['foo'])
        self.assertArrayEqual(data['logical'], actual['logical'])
        dt_vv = np.dtype([('foo', vidt), ('bar', vidt)])
        f.create_dataset('dt_vv', shape=(4,), dtype=dt_vv)
        dt_vvb = np.dtype([('foo', vidt), ('bar', vidt), ('logical', bool)])
        vvb = f.create_dataset('dt_vvb', shape=(2,), dtype=dt_vvb)
        dt_bvv = np.dtype([('logical', bool), ('foo', vidt), ('bar', vidt)])
        bvv = f.create_dataset('dt_bvv', shape=(2,), dtype=dt_bvv)
        data = np.array([(True, a([1, 2, 3]), a([1, 2])), (False, a([]), a([2, 4, 6]))], dtype=bvv)
        bvv[:] = data
        actual = bvv[:]
        self.assertVlenArrayEqual(data['foo'], actual['foo'])
        self.assertVlenArrayEqual(data['bar'], actual['bar'])
        self.assertArrayEqual(data['logical'], actual['logical'])

    def test_compound_vlen_enum(self):
        eidt = h5py.enum_dtype({'OFF': 0, 'ON': 1}, basetype=np.uint8)
        vidt = h5py.vlen_dtype(np.uint8)

        def a(items):
            return np.array(items, dtype=np.uint8)
        f = self.f
        dt_vve = np.dtype([('foo', vidt), ('bar', vidt), ('switch', eidt)])
        vve = f.create_dataset('dt_vve', shape=(2,), dtype=dt_vve)
        data = np.array([(a([1, 2, 3]), a([1, 2]), 1), (a([]), a([2, 4, 6]), 0)], dtype=dt_vve)
        vve[:] = data
        actual = vve[:]
        self.assertVlenArrayEqual(data['foo'], actual['foo'])
        self.assertVlenArrayEqual(data['bar'], actual['bar'])
        self.assertArrayEqual(data['switch'], actual['switch'])

    def test_vlen_enum(self):
        fname = self.mktemp()
        arr1 = [[1], [1, 2]]
        dt1 = h5py.vlen_dtype(h5py.enum_dtype(dict(foo=1, bar=2), 'i'))
        with h5py.File(fname, 'w') as f:
            df1 = f.create_dataset('test', (len(arr1),), dtype=dt1)
            df1[:] = np.array(arr1, dtype=object)
        with h5py.File(fname, 'r') as f:
            df2 = f['test']
            dt2 = df2.dtype
            arr2 = [e.tolist() for e in df2[:]]
        self.assertEqual(arr1, arr2)
        self.assertEqual(h5py.check_enum_dtype(h5py.check_vlen_dtype(dt1)), h5py.check_enum_dtype(h5py.check_vlen_dtype(dt2)))