import unittest as ut
from h5py import h5p, h5f, version
from .common import TestCase
@ut.skipUnless(version.hdf5_version_tuple >= (1, 12, 1) or (version.hdf5_version_tuple[:2] == (1, 10) and version.hdf5_version_tuple[2] >= 7), 'Requires HDF5 1.12.1 or later or 1.10.x >= 1.10.7')
def test_set_file_locking(self):
    """test get/set file locking"""
    falist = h5p.create(h5p.FILE_ACCESS)
    use_file_locking = False
    ignore_when_disabled = False
    falist.set_file_locking(use_file_locking, ignore_when_disabled)
    self.assertEqual((use_file_locking, ignore_when_disabled), falist.get_file_locking())