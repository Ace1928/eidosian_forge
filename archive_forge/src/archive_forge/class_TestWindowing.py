import logging
import unittest
import numpy as np
from gensim import utils
from gensim.test.utils import datapath, get_tmpfile
class TestWindowing(unittest.TestCase):
    arr10_5 = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [5, 6, 7, 8, 9]])

    def _assert_arrays_equal(self, expected, actual):
        self.assertEqual(expected.shape, actual.shape)
        self.assertTrue((actual == expected).all())

    def test_strided_windows1(self):
        out = utils.strided_windows(range(5), 2)
        expected = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
        self._assert_arrays_equal(expected, out)

    def test_strided_windows2(self):
        input_arr = np.arange(10)
        out = utils.strided_windows(input_arr, 5)
        expected = self.arr10_5.copy()
        self._assert_arrays_equal(expected, out)
        out[0, 0] = 10
        self.assertEqual(10, input_arr[0], 'should make view rather than copy')

    def test_strided_windows_window_size_exceeds_size(self):
        input_arr = np.array(['this', 'is', 'test'], dtype='object')
        out = utils.strided_windows(input_arr, 4)
        expected = np.ndarray((0, 0))
        self._assert_arrays_equal(expected, out)

    def test_strided_windows_window_size_equals_size(self):
        input_arr = np.array(['this', 'is', 'test'], dtype='object')
        out = utils.strided_windows(input_arr, 3)
        expected = np.array([input_arr.copy()])
        self._assert_arrays_equal(expected, out)

    def test_iter_windows_include_below_window_size(self):
        texts = [['this', 'is', 'a'], ['test', 'document']]
        out = utils.iter_windows(texts, 3, ignore_below_size=False)
        windows = [list(w) for w in out]
        self.assertEqual(texts, windows)
        out = utils.iter_windows(texts, 3)
        windows = [list(w) for w in out]
        self.assertEqual([texts[0]], windows)

    def test_iter_windows_list_texts(self):
        texts = [['this', 'is', 'a'], ['test', 'document']]
        windows = list(utils.iter_windows(texts, 2))
        list_windows = [list(iterable) for iterable in windows]
        expected = [['this', 'is'], ['is', 'a'], ['test', 'document']]
        self.assertListEqual(list_windows, expected)

    def test_iter_windows_uses_views(self):
        texts = [np.array(['this', 'is', 'a'], dtype='object'), ['test', 'document']]
        windows = list(utils.iter_windows(texts, 2))
        list_windows = [list(iterable) for iterable in windows]
        expected = [['this', 'is'], ['is', 'a'], ['test', 'document']]
        self.assertListEqual(list_windows, expected)
        windows[0][0] = 'modified'
        self.assertEqual('modified', texts[0][0])

    def test_iter_windows_with_copy(self):
        texts = [np.array(['this', 'is', 'a'], dtype='object'), np.array(['test', 'document'], dtype='object')]
        windows = list(utils.iter_windows(texts, 2, copy=True))
        windows[0][0] = 'modified'
        self.assertEqual('this', texts[0][0])
        windows[2][0] = 'modified'
        self.assertEqual('test', texts[1][0])

    def test_flatten_nested(self):
        nested_list = [[[1, 2, 3], [4, 5]], 6]
        expected = [1, 2, 3, 4, 5, 6]
        self.assertEqual(utils.flatten(nested_list), expected)

    def test_flatten_not_nested(self):
        not_nested = [1, 2, 3, 4, 5, 6]
        expected = [1, 2, 3, 4, 5, 6]
        self.assertEqual(utils.flatten(not_nested), expected)