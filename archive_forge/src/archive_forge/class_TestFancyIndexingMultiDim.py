import itertools
import numpy as np
import unittest
from numba import jit, typeof, njit
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import MemoryLeakMixin, TestCase
class TestFancyIndexingMultiDim(MemoryLeakMixin, TestCase):
    shape = (5, 6, 7, 8, 9, 10)
    indexing_cases = [(slice(4, 5), 3, np.array([0, 1, 3, 4, 2]), 1), (3, np.array([0, 1, 3, 4, 2]), slice(None), slice(4)), (Ellipsis, 1, np.array([0, 1, 3, 4, 2])), (np.array([0, 1, 3, 4, 2]), 3, Ellipsis), (Ellipsis, 1, np.array([0, 1, 3, 4, 2]), 3, slice(1, 5)), (np.array([0, 1, 3, 4, 2]), 3, Ellipsis, slice(1, 5)), (slice(4, 5), 3, np.array([True, False, True, False, True, False, False]), 1), (3, np.array([True, False, True, False, True, False]), slice(None), slice(4))]

    def setUp(self):
        super().setUp()
        self.rng = np.random.default_rng(1)

    def generate_random_indices(self):
        N = min(self.shape)
        slice_choices = [slice(None, None, None), slice(1, N - 1, None), slice(0, None, 2), slice(N - 1, None, -2), slice(-N + 1, -1, None), slice(-1, -N, -2), slice(0, N - 1, None), slice(-1, -N, -2)]
        integer_choices = list(np.arange(N))
        indices = []
        K = 20
        for _ in range(K):
            array_idx = self.rng.integers(0, 5, size=15)
            curr_idx = self.rng.choice(slice_choices, size=4).tolist()
            _array_idx = self.rng.choice(4)
            curr_idx[_array_idx] = array_idx
            indices.append(tuple(curr_idx))
        for _ in range(K):
            array_idx = self.rng.integers(0, 5, size=15)
            curr_idx = self.rng.choice(integer_choices, size=4).tolist()
            _array_idx = self.rng.choice(4)
            curr_idx[_array_idx] = array_idx
            indices.append(tuple(curr_idx))
        for _ in range(K):
            array_idx = self.rng.integers(0, 5, size=15)
            curr_idx = self.rng.choice(slice_choices, size=4).tolist()
            _array_idx = self.rng.choice(4, size=2, replace=False)
            curr_idx[_array_idx[0]] = array_idx
            curr_idx[_array_idx[1]] = Ellipsis
            indices.append(tuple(curr_idx))
        for _ in range(K):
            array_idx = self.rng.integers(0, 5, size=15)
            curr_idx = self.rng.choice(slice_choices, size=4).tolist()
            _array_idx = self.rng.choice(4)
            bool_arr_shape = self.shape[_array_idx]
            curr_idx[_array_idx] = np.array(self.rng.choice(2, size=bool_arr_shape), dtype=bool)
            indices.append(tuple(curr_idx))
        return indices

    def check_getitem_indices(self, arr_shape, index):

        @njit
        def numba_get_item(array, idx):
            return array[idx]
        arr = np.random.randint(0, 11, size=arr_shape)
        get_item = numba_get_item.py_func
        orig_base = arr.base or arr
        expected = get_item(arr, index)
        got = numba_get_item(arr, index)
        self.assertIsNot(expected.base, orig_base)
        self.assertEqual(got.shape, expected.shape)
        self.assertEqual(got.dtype, expected.dtype)
        np.testing.assert_equal(got, expected)
        self.assertFalse(np.may_share_memory(got, expected))

    def check_setitem_indices(self, arr_shape, index):

        @njit
        def set_item(array, idx, item):
            array[idx] = item
        arr = np.random.randint(0, 11, size=arr_shape)
        src = arr[index]
        expected = np.zeros_like(arr)
        got = np.zeros_like(arr)
        set_item.py_func(expected, index, src)
        set_item(got, index, src)
        self.assertEqual(got.shape, expected.shape)
        self.assertEqual(got.dtype, expected.dtype)
        np.testing.assert_equal(got, expected)

    def test_getitem(self):
        indices = self.indexing_cases.copy()
        indices += self.generate_random_indices()
        for idx in indices:
            with self.subTest(idx=idx):
                self.check_getitem_indices(self.shape, idx)

    def test_setitem(self):
        indices = self.indexing_cases.copy()
        indices += self.generate_random_indices()
        for idx in indices:
            with self.subTest(idx=idx):
                self.check_setitem_indices(self.shape, idx)

    def test_unsupported_condition_exceptions(self):
        err_idx_cases = [('Multi-dimensional indices are not supported.', (0, 3, np.array([[1, 2], [2, 3]]))), ('Using more than one non-scalar array index is unsupported.', (0, 3, np.array([1, 2]), np.array([1, 2]))), ('Using more than one indexing subspace is unsupported.' + ' An indexing subspace is a group of one or more consecutive' + ' indices comprising integer or array types.', (0, np.array([1, 2]), slice(None), 3, 4))]
        for err, idx in err_idx_cases:
            with self.assertRaises(TypingError) as raises:
                self.check_getitem_indices(self.shape, idx)
            self.assertIn(err, str(raises.exception))