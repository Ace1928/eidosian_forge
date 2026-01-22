import unittest
from fastimport import (
class TestCommonDirectory(unittest.TestCase):

    def test_no_paths(self):
        c = helpers.common_directory(None)
        self.assertEqual(c, None)
        c = helpers.common_directory([])
        self.assertEqual(c, None)

    def test_one_path(self):
        c = helpers.common_directory([b'foo'])
        self.assertEqual(c, b'')
        c = helpers.common_directory([b'foo/'])
        self.assertEqual(c, b'foo/')
        c = helpers.common_directory([b'foo/bar'])
        self.assertEqual(c, b'foo/')

    def test_two_paths(self):
        c = helpers.common_directory([b'foo', b'bar'])
        self.assertEqual(c, b'')
        c = helpers.common_directory([b'foo/', b'bar'])
        self.assertEqual(c, b'')
        c = helpers.common_directory([b'foo/', b'foo/bar'])
        self.assertEqual(c, b'foo/')
        c = helpers.common_directory([b'foo/bar/x', b'foo/bar/y'])
        self.assertEqual(c, b'foo/bar/')
        c = helpers.common_directory([b'foo/bar/aa_x', b'foo/bar/aa_y'])
        self.assertEqual(c, b'foo/bar/')

    def test_lots_of_paths(self):
        c = helpers.common_directory([b'foo/bar/x', b'foo/bar/y', b'foo/bar/z'])
        self.assertEqual(c, b'foo/bar/')