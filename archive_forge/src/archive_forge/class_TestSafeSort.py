from collections import defaultdict
from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
import pandas.core.common as com
from pandas.core.sorting import (
class TestSafeSort:

    @pytest.mark.parametrize('arg, exp', [[[3, 1, 2, 0, 4], [0, 1, 2, 3, 4]], [np.array(list('baaacb'), dtype=object), np.array(list('aaabbc'), dtype=object)], [[], []]])
    def test_basic_sort(self, arg, exp):
        result = safe_sort(np.array(arg))
        expected = np.array(exp)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('verify', [True, False])
    @pytest.mark.parametrize('codes, exp_codes', [[[0, 1, 1, 2, 3, 0, -1, 4], [3, 1, 1, 2, 0, 3, -1, 4]], [[], []]])
    def test_codes(self, verify, codes, exp_codes):
        values = np.array([3, 1, 2, 0, 4])
        expected = np.array([0, 1, 2, 3, 4])
        result, result_codes = safe_sort(values, codes, use_na_sentinel=True, verify=verify)
        expected_codes = np.array(exp_codes, dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)
        tm.assert_numpy_array_equal(result_codes, expected_codes)

    def test_codes_out_of_bound(self):
        values = np.array([3, 1, 2, 0, 4])
        expected = np.array([0, 1, 2, 3, 4])
        codes = [0, 101, 102, 2, 3, 0, 99, 4]
        result, result_codes = safe_sort(values, codes, use_na_sentinel=True)
        expected_codes = np.array([3, -1, -1, 2, 0, 3, -1, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)
        tm.assert_numpy_array_equal(result_codes, expected_codes)

    def test_mixed_integer(self):
        values = np.array(['b', 1, 0, 'a', 0, 'b'], dtype=object)
        result = safe_sort(values)
        expected = np.array([0, 0, 1, 'a', 'b', 'b'], dtype=object)
        tm.assert_numpy_array_equal(result, expected)

    def test_mixed_integer_with_codes(self):
        values = np.array(['b', 1, 0, 'a'], dtype=object)
        codes = [0, 1, 2, 3, 0, -1, 1]
        result, result_codes = safe_sort(values, codes)
        expected = np.array([0, 1, 'a', 'b'], dtype=object)
        expected_codes = np.array([3, 1, 0, 2, 3, -1, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)
        tm.assert_numpy_array_equal(result_codes, expected_codes)

    def test_unsortable(self):
        arr = np.array([1, 2, datetime.now(), 0, 3], dtype=object)
        msg = "'[<>]' not supported between instances of .*"
        with pytest.raises(TypeError, match=msg):
            safe_sort(arr)

    @pytest.mark.parametrize('arg, codes, err, msg', [[1, None, TypeError, 'Only np.ndarray, ExtensionArray, and Index'], [np.array([0, 1, 2]), 1, TypeError, 'Only list-like objects or None'], [np.array([0, 1, 2, 1]), [0, 1], ValueError, 'values should be unique']])
    def test_exceptions(self, arg, codes, err, msg):
        with pytest.raises(err, match=msg):
            safe_sort(values=arg, codes=codes)

    @pytest.mark.parametrize('arg, exp', [[[1, 3, 2], [1, 2, 3]], [[1, 3, np.nan, 2], [1, 2, 3, np.nan]]])
    def test_extension_array(self, arg, exp):
        a = array(arg, dtype='Int64')
        result = safe_sort(a)
        expected = array(exp, dtype='Int64')
        tm.assert_extension_array_equal(result, expected)

    @pytest.mark.parametrize('verify', [True, False])
    def test_extension_array_codes(self, verify):
        a = array([1, 3, 2], dtype='Int64')
        result, codes = safe_sort(a, [0, 1, -1, 2], use_na_sentinel=True, verify=verify)
        expected_values = array([1, 2, 3], dtype='Int64')
        expected_codes = np.array([0, 2, -1, 1], dtype=np.intp)
        tm.assert_extension_array_equal(result, expected_values)
        tm.assert_numpy_array_equal(codes, expected_codes)