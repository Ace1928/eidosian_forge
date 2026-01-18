import numpy as np
from .common import TestCase, ut
import h5py
from h5py import h5a, h5s, h5t
from h5py import File
from h5py._hl.base import is_empty_dataspace
def test_string_scalar(self):
    """ Storage of variable-length byte string scalars (auto-creation) """
    self.f.attrs['x'] = b'Hello'
    out = self.f.attrs['x']
    self.assertEqual(out, 'Hello')
    self.assertEqual(type(out), str)
    aid = h5py.h5a.open(self.f.id, b'x')
    tid = aid.get_type()
    self.assertEqual(type(tid), h5py.h5t.TypeStringID)
    self.assertEqual(tid.get_cset(), h5py.h5t.CSET_ASCII)
    self.assertTrue(tid.is_variable_str())