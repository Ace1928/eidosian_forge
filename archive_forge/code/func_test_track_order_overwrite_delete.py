import numpy as np
from collections.abc import MutableMapping
from .common import TestCase, ut
import h5py
from h5py import File
from h5py import h5a,  h5t
from h5py import AttributeManager
@ut.skipUnless(h5py.version.hdf5_version_tuple >= (1, 10, 6), 'HDF5 1.10.6 required')
def test_track_order_overwrite_delete(self):
    group = self.fill_attrs2(track_order=True)
    self.assertEqual(group.attrs['11'], 11)
    group.attrs['11'] = 42.0
    self.assertEqual(group.attrs['11'], 42.0)
    self.assertIn('10', group.attrs)
    del group.attrs['10']
    self.assertNotIn('10', group.attrs)