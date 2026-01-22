import numpy as np
from .common import TestCase, ut
import h5py
from h5py import h5a, h5s, h5t
from h5py import File
from h5py._hl.base import is_empty_dataspace
class TestEmpty(BaseAttrs):

    def setUp(self):
        BaseAttrs.setUp(self)
        sid = h5s.create(h5s.NULL)
        tid = h5t.C_S1.copy()
        tid.set_size(10)
        aid = h5a.create(self.f.id, b'x', tid, sid)
        self.empty_obj = h5py.Empty(np.dtype('S10'))

    def test_read(self):
        self.assertEqual(self.empty_obj, self.f.attrs['x'])

    def test_write(self):
        self.f.attrs['y'] = self.empty_obj
        self.assertTrue(is_empty_dataspace(h5a.open(self.f.id, b'y')))

    def test_modify(self):
        with self.assertRaises(IOError):
            self.f.attrs.modify('x', 1)

    def test_values(self):
        values = list(self.f.attrs.values())
        self.assertEqual([self.empty_obj], values)

    def test_items(self):
        items = list(self.f.attrs.items())
        self.assertEqual([(u'x', self.empty_obj)], items)

    def test_itervalues(self):
        values = list(self.f.attrs.values())
        self.assertEqual([self.empty_obj], values)

    def test_iteritems(self):
        items = list(self.f.attrs.items())
        self.assertEqual([(u'x', self.empty_obj)], items)