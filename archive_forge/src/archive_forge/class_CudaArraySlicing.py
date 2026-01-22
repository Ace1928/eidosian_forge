from itertools import product
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from unittest.mock import patch
class CudaArraySlicing(CUDATestCase):

    def test_prefix_1d(self):
        arr = np.arange(5)
        darr = cuda.to_device(arr)
        for i in range(arr.size):
            expect = arr[i:]
            got = darr[i:].copy_to_host()
            self.assertTrue(np.all(expect == got))

    def test_prefix_2d(self):
        arr = np.arange(3 ** 2).reshape(3, 3)
        darr = cuda.to_device(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                expect = arr[i:, j:]
                sliced = darr[i:, j:]
                self.assertEqual(expect.shape, sliced.shape)
                self.assertEqual(expect.strides, sliced.strides)
                got = sliced.copy_to_host()
                self.assertTrue(np.all(expect == got))

    def test_select_3d_first_two_dim(self):
        arr = np.arange(3 * 4 * 5).reshape(3, 4, 5)
        darr = cuda.to_device(arr)
        for i in range(arr.shape[0]):
            expect = arr[i]
            sliced = darr[i]
            self.assertEqual(expect.shape, sliced.shape)
            self.assertEqual(expect.strides, sliced.strides)
            got = sliced.copy_to_host()
            self.assertTrue(np.all(expect == got))
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                expect = arr[i, j]
                sliced = darr[i, j]
                self.assertEqual(expect.shape, sliced.shape)
                self.assertEqual(expect.strides, sliced.strides)
                got = sliced.copy_to_host()
                self.assertTrue(np.all(expect == got))

    def test_select_f(self):
        a = np.arange(5 * 6 * 7).reshape(5, 6, 7, order='F')
        da = cuda.to_device(a)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                self.assertTrue(np.array_equal(da[i, j, :].copy_to_host(), a[i, j, :]))
            for j in range(a.shape[2]):
                self.assertTrue(np.array_equal(da[i, :, j].copy_to_host(), a[i, :, j]))
        for i in range(a.shape[1]):
            for j in range(a.shape[2]):
                self.assertTrue(np.array_equal(da[:, i, j].copy_to_host(), a[:, i, j]))

    def test_select_c(self):
        a = np.arange(5 * 6 * 7).reshape(5, 6, 7, order='C')
        da = cuda.to_device(a)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                self.assertTrue(np.array_equal(da[i, j, :].copy_to_host(), a[i, j, :]))
            for j in range(a.shape[2]):
                self.assertTrue(np.array_equal(da[i, :, j].copy_to_host(), a[i, :, j]))
        for i in range(a.shape[1]):
            for j in range(a.shape[2]):
                self.assertTrue(np.array_equal(da[:, i, j].copy_to_host(), a[:, i, j]))

    def test_prefix_select(self):
        arr = np.arange(5 * 7).reshape(5, 7, order='F')
        darr = cuda.to_device(arr)
        self.assertTrue(np.all(darr[:1, 1].copy_to_host() == arr[:1, 1]))

    def test_negative_slicing_1d(self):
        arr = np.arange(10)
        darr = cuda.to_device(arr)
        for i, j in product(range(-10, 10), repeat=2):
            np.testing.assert_array_equal(arr[i:j], darr[i:j].copy_to_host())

    def test_negative_slicing_2d(self):
        arr = np.arange(12).reshape(3, 4)
        darr = cuda.to_device(arr)
        for x, y, w, s in product(range(-4, 4), repeat=4):
            np.testing.assert_array_equal(arr[x:y, w:s], darr[x:y, w:s].copy_to_host())

    def test_empty_slice_1d(self):
        arr = np.arange(5)
        darr = cuda.to_device(arr)
        for i in range(darr.shape[0]):
            np.testing.assert_array_equal(darr[i:i].copy_to_host(), arr[i:i])
        self.assertFalse(darr[:0][:0].copy_to_host())
        np.testing.assert_array_equal(darr[:0][:1].copy_to_host(), arr[:0][:1])
        np.testing.assert_array_equal(darr[:0][-1:].copy_to_host(), arr[:0][-1:])

    def test_empty_slice_2d(self):
        arr = np.arange(5 * 7).reshape(5, 7)
        darr = cuda.to_device(arr)
        np.testing.assert_array_equal(darr[:0].copy_to_host(), arr[:0])
        np.testing.assert_array_equal(darr[3, :0].copy_to_host(), arr[3, :0])
        self.assertFalse(darr[:0][:0].copy_to_host())
        np.testing.assert_array_equal(darr[:0][:1].copy_to_host(), arr[:0][:1])
        np.testing.assert_array_equal(darr[:0][-1:].copy_to_host(), arr[:0][-1:])