import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
class TestPadWidth:

    @pytest.mark.parametrize('pad_width', [(4, 5, 6, 7), ((1,), (2,), (3,)), ((1, 2), (3, 4), (5, 6)), ((3, 4, 5), (0, 1, 2))])
    @pytest.mark.parametrize('mode', _all_modes.keys())
    def test_misshaped_pad_width(self, pad_width, mode):
        arr = np.arange(30).reshape((6, 5))
        match = 'operands could not be broadcast together'
        with pytest.raises(ValueError, match=match):
            np.pad(arr, pad_width, mode)

    @pytest.mark.parametrize('mode', _all_modes.keys())
    def test_misshaped_pad_width_2(self, mode):
        arr = np.arange(30).reshape((6, 5))
        match = 'input operand has more dimensions than allowed by the axis remapping'
        with pytest.raises(ValueError, match=match):
            np.pad(arr, (((3,), (4,), (5,)), ((0,), (1,), (2,))), mode)

    @pytest.mark.parametrize('pad_width', [-2, (-2,), (3, -1), ((5, 2), (-2, 3)), ((-4,), (2,))])
    @pytest.mark.parametrize('mode', _all_modes.keys())
    def test_negative_pad_width(self, pad_width, mode):
        arr = np.arange(30).reshape((6, 5))
        match = "index can't contain negative values"
        with pytest.raises(ValueError, match=match):
            np.pad(arr, pad_width, mode)

    @pytest.mark.parametrize('pad_width, dtype', [('3', None), ('word', None), (None, None), (object(), None), (3.4, None), (((2, 3, 4), (3, 2)), object), (complex(1, -1), None), (((-2.1, 3), (3, 2)), None)])
    @pytest.mark.parametrize('mode', _all_modes.keys())
    def test_bad_type(self, pad_width, dtype, mode):
        arr = np.arange(30).reshape((6, 5))
        match = '`pad_width` must be of integral type.'
        if dtype is not None:
            with pytest.raises(TypeError, match=match):
                np.pad(arr, np.array(pad_width, dtype=dtype), mode)
        else:
            with pytest.raises(TypeError, match=match):
                np.pad(arr, pad_width, mode)
            with pytest.raises(TypeError, match=match):
                np.pad(arr, np.array(pad_width), mode)

    def test_pad_width_as_ndarray(self):
        a = np.arange(12)
        a = np.reshape(a, (4, 3))
        a = np.pad(a, np.array(((2, 3), (3, 2))), 'edge')
        b = np.array([[0, 0, 0, 0, 1, 2, 2, 2], [0, 0, 0, 0, 1, 2, 2, 2], [0, 0, 0, 0, 1, 2, 2, 2], [3, 3, 3, 3, 4, 5, 5, 5], [6, 6, 6, 6, 7, 8, 8, 8], [9, 9, 9, 9, 10, 11, 11, 11], [9, 9, 9, 9, 10, 11, 11, 11], [9, 9, 9, 9, 10, 11, 11, 11], [9, 9, 9, 9, 10, 11, 11, 11]])
        assert_array_equal(a, b)

    @pytest.mark.parametrize('pad_width', [0, (0, 0), ((0, 0), (0, 0))])
    @pytest.mark.parametrize('mode', _all_modes.keys())
    def test_zero_pad_width(self, pad_width, mode):
        arr = np.arange(30).reshape(6, 5)
        assert_array_equal(arr, np.pad(arr, pad_width, mode=mode))