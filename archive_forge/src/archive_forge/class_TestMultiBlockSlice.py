import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
class TestMultiBlockSlice(BaseSlicing):

    def setUp(self):
        super().setUp()
        self.arr = np.arange(10)
        self.dset = self.f.create_dataset('x', data=self.arr)

    def test_default(self):
        mbslice = MultiBlockSlice()
        self.assertEqual(mbslice.indices(10), (0, 1, 10, 1))
        np.testing.assert_array_equal(self.dset[mbslice], self.arr)

    def test_default_explicit(self):
        mbslice = MultiBlockSlice(start=0, count=10, stride=1, block=1)
        self.assertEqual(mbslice.indices(10), (0, 1, 10, 1))
        np.testing.assert_array_equal(self.dset[mbslice], self.arr)

    def test_start(self):
        mbslice = MultiBlockSlice(start=4)
        self.assertEqual(mbslice.indices(10), (4, 1, 6, 1))
        np.testing.assert_array_equal(self.dset[mbslice], np.array([4, 5, 6, 7, 8, 9]))

    def test_count(self):
        mbslice = MultiBlockSlice(count=7)
        self.assertEqual(mbslice.indices(10), (0, 1, 7, 1))
        np.testing.assert_array_equal(self.dset[mbslice], np.array([0, 1, 2, 3, 4, 5, 6]))

    def test_count_more_than_length_error(self):
        mbslice = MultiBlockSlice(count=11)
        with self.assertRaises(ValueError):
            mbslice.indices(10)

    def test_stride(self):
        mbslice = MultiBlockSlice(stride=2)
        self.assertEqual(mbslice.indices(10), (0, 2, 5, 1))
        np.testing.assert_array_equal(self.dset[mbslice], np.array([0, 2, 4, 6, 8]))

    def test_stride_zero_error(self):
        with self.assertRaises(ValueError):
            MultiBlockSlice(stride=0, block=0).indices(10)

    def test_stride_block_equal(self):
        mbslice = MultiBlockSlice(stride=2, block=2)
        self.assertEqual(mbslice.indices(10), (0, 2, 5, 2))
        np.testing.assert_array_equal(self.dset[mbslice], self.arr)

    def test_block_more_than_stride_error(self):
        with self.assertRaises(ValueError):
            MultiBlockSlice(block=3)
        with self.assertRaises(ValueError):
            MultiBlockSlice(stride=2, block=3)

    def test_stride_more_than_block(self):
        mbslice = MultiBlockSlice(stride=3, block=2)
        self.assertEqual(mbslice.indices(10), (0, 3, 3, 2))
        np.testing.assert_array_equal(self.dset[mbslice], np.array([0, 1, 3, 4, 6, 7]))

    def test_block_overruns_extent_error(self):
        mbslice = MultiBlockSlice(start=2, count=2, stride=5, block=4)
        with self.assertRaises(ValueError):
            mbslice.indices(10)

    def test_fully_described(self):
        mbslice = MultiBlockSlice(start=1, count=2, stride=5, block=4)
        self.assertEqual(mbslice.indices(10), (1, 5, 2, 4))
        np.testing.assert_array_equal(self.dset[mbslice], np.array([1, 2, 3, 4, 6, 7, 8, 9]))

    def test_count_calculated(self):
        mbslice = MultiBlockSlice(start=1, stride=3, block=2)
        self.assertEqual(mbslice.indices(10), (1, 3, 3, 2))
        np.testing.assert_array_equal(self.dset[mbslice], np.array([1, 2, 4, 5, 7, 8]))

    def test_zero_count_calculated_error(self):
        mbslice = MultiBlockSlice(start=8, stride=4, block=3)
        with self.assertRaises(ValueError):
            mbslice.indices(10)