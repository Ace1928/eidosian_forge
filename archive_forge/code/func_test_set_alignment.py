import unittest as ut
from h5py import h5p, h5f, version
from .common import TestCase
def test_set_alignment(self):
    """test get/set chunk cache """
    falist = h5p.create(h5p.FILE_ACCESS)
    threshold = 10 * 1024
    alignment = 1024 * 1024
    falist.set_alignment(threshold, alignment)
    self.assertEqual((threshold, alignment), falist.get_alignment())