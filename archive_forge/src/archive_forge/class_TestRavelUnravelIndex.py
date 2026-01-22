import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
class TestRavelUnravelIndex:

    def test_basic(self):
        assert_equal(np.unravel_index(2, (2, 2)), (1, 0))
        assert_equal(np.unravel_index(indices=2, shape=(2, 2)), (1, 0))
        with assert_raises(TypeError):
            np.unravel_index(indices=2, hape=(2, 2))
        with assert_raises(TypeError):
            np.unravel_index(2, hape=(2, 2))
        with assert_raises(TypeError):
            np.unravel_index(254, ims=(17, 94))
        with assert_raises(TypeError):
            np.unravel_index(254, dims=(17, 94))
        assert_equal(np.ravel_multi_index((1, 0), (2, 2)), 2)
        assert_equal(np.unravel_index(254, (17, 94)), (2, 66))
        assert_equal(np.ravel_multi_index((2, 66), (17, 94)), 254)
        assert_raises(ValueError, np.unravel_index, -1, (2, 2))
        assert_raises(TypeError, np.unravel_index, 0.5, (2, 2))
        assert_raises(ValueError, np.unravel_index, 4, (2, 2))
        assert_raises(ValueError, np.ravel_multi_index, (-3, 1), (2, 2))
        assert_raises(ValueError, np.ravel_multi_index, (2, 1), (2, 2))
        assert_raises(ValueError, np.ravel_multi_index, (0, -3), (2, 2))
        assert_raises(ValueError, np.ravel_multi_index, (0, 2), (2, 2))
        assert_raises(TypeError, np.ravel_multi_index, (0.1, 0.0), (2, 2))
        assert_equal(np.unravel_index((2 * 3 + 1) * 6 + 4, (4, 3, 6)), [2, 1, 4])
        assert_equal(np.ravel_multi_index([2, 1, 4], (4, 3, 6)), (2 * 3 + 1) * 6 + 4)
        arr = np.array([[3, 6, 6], [4, 5, 1]])
        assert_equal(np.ravel_multi_index(arr, (7, 6)), [22, 41, 37])
        assert_equal(np.ravel_multi_index(arr, (7, 6), order='F'), [31, 41, 13])
        assert_equal(np.ravel_multi_index(arr, (4, 6), mode='clip'), [22, 23, 19])
        assert_equal(np.ravel_multi_index(arr, (4, 4), mode=('clip', 'wrap')), [12, 13, 13])
        assert_equal(np.ravel_multi_index((3, 1, 4, 1), (6, 7, 8, 9)), 1621)
        assert_equal(np.unravel_index(np.array([22, 41, 37]), (7, 6)), [[3, 6, 6], [4, 5, 1]])
        assert_equal(np.unravel_index(np.array([31, 41, 13]), (7, 6), order='F'), [[3, 6, 6], [4, 5, 1]])
        assert_equal(np.unravel_index(1621, (6, 7, 8, 9)), [3, 1, 4, 1])

    def test_empty_indices(self):
        msg1 = 'indices must be integral: the provided empty sequence was'
        msg2 = 'only int indices permitted'
        assert_raises_regex(TypeError, msg1, np.unravel_index, [], (10, 3, 5))
        assert_raises_regex(TypeError, msg1, np.unravel_index, (), (10, 3, 5))
        assert_raises_regex(TypeError, msg2, np.unravel_index, np.array([]), (10, 3, 5))
        assert_equal(np.unravel_index(np.array([], dtype=int), (10, 3, 5)), [[], [], []])
        assert_raises_regex(TypeError, msg1, np.ravel_multi_index, ([], []), (10, 3))
        assert_raises_regex(TypeError, msg1, np.ravel_multi_index, ([], ['abc']), (10, 3))
        assert_raises_regex(TypeError, msg2, np.ravel_multi_index, (np.array([]), np.array([])), (5, 3))
        assert_equal(np.ravel_multi_index((np.array([], dtype=int), np.array([], dtype=int)), (5, 3)), [])
        assert_equal(np.ravel_multi_index(np.array([[], []], dtype=int), (5, 3)), [])

    def test_big_indices(self):
        if np.intp == np.int64:
            arr = ([1, 29], [3, 5], [3, 117], [19, 2], [2379, 1284], [2, 2], [0, 1])
            assert_equal(np.ravel_multi_index(arr, (41, 7, 120, 36, 2706, 8, 6)), [5627771580, 117259570957])
        assert_raises(ValueError, np.unravel_index, 1, (2 ** 32 - 1, 2 ** 31 + 1))
        dummy_arr = ([0], [0])
        half_max = np.iinfo(np.intp).max // 2
        assert_equal(np.ravel_multi_index(dummy_arr, (half_max, 2)), [0])
        assert_raises(ValueError, np.ravel_multi_index, dummy_arr, (half_max + 1, 2))
        assert_equal(np.ravel_multi_index(dummy_arr, (half_max, 2), order='F'), [0])
        assert_raises(ValueError, np.ravel_multi_index, dummy_arr, (half_max + 1, 2), order='F')

    def test_dtypes(self):
        for dtype in [np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]:
            coords = np.array([[1, 0, 1, 2, 3, 4], [1, 6, 1, 3, 2, 0]], dtype=dtype)
            shape = (5, 8)
            uncoords = 8 * coords[0] + coords[1]
            assert_equal(np.ravel_multi_index(coords, shape), uncoords)
            assert_equal(coords, np.unravel_index(uncoords, shape))
            uncoords = coords[0] + 5 * coords[1]
            assert_equal(np.ravel_multi_index(coords, shape, order='F'), uncoords)
            assert_equal(coords, np.unravel_index(uncoords, shape, order='F'))
            coords = np.array([[1, 0, 1, 2, 3, 4], [1, 6, 1, 3, 2, 0], [1, 3, 1, 0, 9, 5]], dtype=dtype)
            shape = (5, 8, 10)
            uncoords = 10 * (8 * coords[0] + coords[1]) + coords[2]
            assert_equal(np.ravel_multi_index(coords, shape), uncoords)
            assert_equal(coords, np.unravel_index(uncoords, shape))
            uncoords = coords[0] + 5 * (coords[1] + 8 * coords[2])
            assert_equal(np.ravel_multi_index(coords, shape, order='F'), uncoords)
            assert_equal(coords, np.unravel_index(uncoords, shape, order='F'))

    def test_clipmodes(self):
        assert_equal(np.ravel_multi_index([5, 1, -1, 2], (4, 3, 7, 12), mode='wrap'), np.ravel_multi_index([1, 1, 6, 2], (4, 3, 7, 12)))
        assert_equal(np.ravel_multi_index([5, 1, -1, 2], (4, 3, 7, 12), mode=('wrap', 'raise', 'clip', 'raise')), np.ravel_multi_index([1, 1, 0, 2], (4, 3, 7, 12)))
        assert_raises(ValueError, np.ravel_multi_index, [5, 1, -1, 2], (4, 3, 7, 12))

    def test_writeability(self):
        x, y = np.unravel_index([1, 2, 3], (4, 5))
        assert_(x.flags.writeable)
        assert_(y.flags.writeable)

    def test_0d(self):
        x = np.unravel_index(0, ())
        assert_equal(x, ())
        assert_raises_regex(ValueError, '0d array', np.unravel_index, [0], ())
        assert_raises_regex(ValueError, 'out of bounds', np.unravel_index, [1], ())

    @pytest.mark.parametrize('mode', ['clip', 'wrap', 'raise'])
    def test_empty_array_ravel(self, mode):
        res = np.ravel_multi_index(np.zeros((3, 0), dtype=np.intp), (2, 1, 0), mode=mode)
        assert res.shape == (0,)
        with assert_raises(ValueError):
            np.ravel_multi_index(np.zeros((3, 1), dtype=np.intp), (2, 1, 0), mode=mode)

    def test_empty_array_unravel(self):
        res = np.unravel_index(np.zeros(0, dtype=np.intp), (2, 1, 0))
        assert len(res) == 3
        assert all((a.shape == (0,) for a in res))
        with assert_raises(ValueError):
            np.unravel_index([1], (2, 1, 0))