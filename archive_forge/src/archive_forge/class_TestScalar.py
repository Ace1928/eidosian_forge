import numpy as np
from .common import TestCase, ut
import h5py
from h5py import h5a, h5s, h5t
from h5py import File
from h5py._hl.base import is_empty_dataspace
class TestScalar(BaseAttrs):
    """
        Feature: Scalar types map correctly to array scalars
    """

    def test_int(self):
        """ Integers are read as correct NumPy type """
        self.f.attrs['x'] = np.array(1, dtype=np.int8)
        out = self.f.attrs['x']
        self.assertIsInstance(out, np.int8)

    def test_compound(self):
        """ Compound scalars are read as numpy.void """
        dt = np.dtype([('a', 'i'), ('b', 'f')])
        data = np.array((1, 4.2), dtype=dt)
        self.f.attrs['x'] = data
        out = self.f.attrs['x']
        self.assertIsInstance(out, np.void)
        self.assertEqual(out, data)
        self.assertEqual(out['b'], data['b'])

    def test_compound_with_vlen_fields(self):
        """ Compound scalars with vlen fields can be written and read """
        dt = np.dtype([('a', h5py.vlen_dtype(np.int32)), ('b', h5py.vlen_dtype(np.int32))])
        data = np.array((np.array(list(range(1, 5)), dtype=np.int32), np.array(list(range(8, 10)), dtype=np.int32)), dtype=dt)[()]
        self.f.attrs['x'] = data
        out = self.f.attrs['x']
        self.assertArrayEqual(out, data, check_alignment=False)

    def test_nesting_compound_with_vlen_fields(self):
        """ Compound scalars with nested compound vlen fields can be written and read """
        dt_inner = np.dtype([('a', h5py.vlen_dtype(np.int32)), ('b', h5py.vlen_dtype(np.int32))])
        dt = np.dtype([('f1', h5py.vlen_dtype(dt_inner)), ('f2', np.int64)])
        inner1 = (np.array(range(1, 3), dtype=np.int32), np.array(range(6, 9), dtype=np.int32))
        inner2 = (np.array(range(10, 14), dtype=np.int32), np.array(range(16, 20), dtype=np.int32))
        data = np.array((np.array([inner1, inner2], dtype=dt_inner), 2), dtype=dt)[()]
        self.f.attrs['x'] = data
        out = self.f.attrs['x']
        self.assertArrayEqual(out, data, check_alignment=False)

    def test_vlen_compound_with_vlen_string(self):
        """ Compound scalars with vlen compounds containing vlen strings can be written and read """
        dt_inner = np.dtype([('a', h5py.string_dtype()), ('b', h5py.string_dtype())])
        dt = np.dtype([('f', h5py.vlen_dtype(dt_inner))])
        data = np.array((np.array([(b'apples', b'bananas'), (b'peaches', b'oranges')], dtype=dt_inner),), dtype=dt)[()]
        self.f.attrs['x'] = data
        out = self.f.attrs['x']
        self.assertArrayEqual(out, data, check_alignment=False)