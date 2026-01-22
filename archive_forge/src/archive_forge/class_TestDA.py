import unittest as ut
from h5py import h5p, h5f, version
from .common import TestCase
class TestDA(TestCase):
    """
    Feature: setting/getting chunk cache size on a dataset access property list
    """

    def test_chunk_cache(self):
        """test get/set chunk cache """
        dalist = h5p.create(h5p.DATASET_ACCESS)
        nslots = 10000
        nbytes = 1000000
        w0 = 0.5
        dalist.set_chunk_cache(nslots, nbytes, w0)
        self.assertEqual((nslots, nbytes, w0), dalist.get_chunk_cache())

    def test_efile_prefix(self):
        """test get/set efile prefix """
        dalist = h5p.create(h5p.DATASET_ACCESS)
        self.assertEqual(dalist.get_efile_prefix().decode(), '')
        efile_prefix = 'path/to/external/dataset'
        dalist.set_efile_prefix(efile_prefix.encode('utf-8'))
        self.assertEqual(dalist.get_efile_prefix().decode(), efile_prefix)
        efile_prefix = '${ORIGIN}'
        dalist.set_efile_prefix(efile_prefix.encode('utf-8'))
        self.assertEqual(dalist.get_efile_prefix().decode(), efile_prefix)

    def test_virtual_prefix(self):
        """test get/set virtual prefix """
        dalist = h5p.create(h5p.DATASET_ACCESS)
        self.assertEqual(dalist.get_virtual_prefix().decode(), '')
        virtual_prefix = 'path/to/virtual/dataset'
        dalist.set_virtual_prefix(virtual_prefix.encode('utf-8'))
        self.assertEqual(dalist.get_virtual_prefix().decode(), virtual_prefix)