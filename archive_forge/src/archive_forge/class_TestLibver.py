import unittest as ut
from h5py import h5p, h5f, version
from .common import TestCase
class TestLibver(TestCase):
    """
        Feature: Setting/getting lib ver bounds
    """

    def test_libver(self):
        """ Test libver bounds set/get """
        plist = h5p.create(h5p.FILE_ACCESS)
        plist.set_libver_bounds(h5f.LIBVER_EARLIEST, h5f.LIBVER_LATEST)
        self.assertEqual((h5f.LIBVER_EARLIEST, h5f.LIBVER_LATEST), plist.get_libver_bounds())

    def test_libver_v18(self):
        """ Test libver bounds set/get for H5F_LIBVER_V18"""
        plist = h5p.create(h5p.FILE_ACCESS)
        plist.set_libver_bounds(h5f.LIBVER_EARLIEST, h5f.LIBVER_V18)
        self.assertEqual((h5f.LIBVER_EARLIEST, h5f.LIBVER_V18), plist.get_libver_bounds())

    def test_libver_v110(self):
        """ Test libver bounds set/get for H5F_LIBVER_V110"""
        plist = h5p.create(h5p.FILE_ACCESS)
        plist.set_libver_bounds(h5f.LIBVER_V18, h5f.LIBVER_V110)
        self.assertEqual((h5f.LIBVER_V18, h5f.LIBVER_V110), plist.get_libver_bounds())

    @ut.skipIf(version.hdf5_version_tuple < (1, 11, 4), 'Requires HDF5 1.11.4 or later')
    def test_libver_v112(self):
        """ Test libver bounds set/get for H5F_LIBVER_V112"""
        plist = h5p.create(h5p.FILE_ACCESS)
        plist.set_libver_bounds(h5f.LIBVER_V18, h5f.LIBVER_V112)
        self.assertEqual((h5f.LIBVER_V18, h5f.LIBVER_V112), plist.get_libver_bounds())