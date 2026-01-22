import collections
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestCategoricalMissing:

    def test_isna(self):
        exp = np.array([False, False, True])
        cat = Categorical(['a', 'b', np.nan])
        res = cat.isna()
        tm.assert_numpy_array_equal(res, exp)

    def test_na_flags_int_categories(self):
        categories = list(range(10))
        labels = np.random.default_rng(2).integers(0, 10, 20)
        labels[::5] = -1
        cat = Categorical(labels, categories)
        repr(cat)
        tm.assert_numpy_array_equal(isna(cat), labels == -1)

    def test_nan_handling(self):
        c = Categorical(['a', 'b', np.nan, 'a'])
        tm.assert_index_equal(c.categories, Index(['a', 'b']))
        tm.assert_numpy_array_equal(c._codes, np.array([0, 1, -1, 0], dtype=np.int8))
        c[1] = np.nan
        tm.assert_index_equal(c.categories, Index(['a', 'b']))
        tm.assert_numpy_array_equal(c._codes, np.array([0, -1, -1, 0], dtype=np.int8))
        c = Categorical(['a', 'b', np.nan, 'a'])
        tm.assert_index_equal(c.categories, Index(['a', 'b']))
        tm.assert_numpy_array_equal(c._codes, np.array([0, 1, -1, 0], dtype=np.int8))

    def test_set_dtype_nans(self):
        c = Categorical(['a', 'b', np.nan])
        result = c._set_dtype(CategoricalDtype(['a', 'c']))
        tm.assert_numpy_array_equal(result.codes, np.array([0, -1, -1], dtype='int8'))

    def test_set_item_nan(self):
        cat = Categorical([1, 2, 3])
        cat[1] = np.nan
        exp = Categorical([1, np.nan, 3], categories=[1, 2, 3])
        tm.assert_categorical_equal(cat, exp)

    @pytest.mark.parametrize('fillna_kwargs, msg', [({'value': 1, 'method': 'ffill'}, "Cannot specify both 'value' and 'method'."), ({}, "Must specify a fill 'value' or 'method'."), ({'method': 'bad'}, 'Invalid fill method. Expecting .* bad'), ({'value': Series([1, 2, 3, 4, 'a'])}, 'Cannot setitem on a Categorical with a new category')])
    def test_fillna_raises(self, fillna_kwargs, msg):
        cat = Categorical([1, 2, 3, None, None])
        if len(fillna_kwargs) == 1 and 'value' in fillna_kwargs:
            err = TypeError
        else:
            err = ValueError
        with pytest.raises(err, match=msg):
            cat.fillna(**fillna_kwargs)

    @pytest.mark.parametrize('named', [True, False])
    def test_fillna_iterable_category(self, named):
        if named:
            Point = collections.namedtuple('Point', 'x y')
        else:
            Point = lambda *args: args
        cat = Categorical(np.array([Point(0, 0), Point(0, 1), None], dtype=object))
        result = cat.fillna(Point(0, 0))
        expected = Categorical([Point(0, 0), Point(0, 1), Point(0, 0)])
        tm.assert_categorical_equal(result, expected)
        cat = Categorical(np.array([Point(1, 0), Point(0, 1), None], dtype=object))
        msg = 'Cannot setitem on a Categorical with a new category'
        with pytest.raises(TypeError, match=msg):
            cat.fillna(Point(0, 0))

    def test_fillna_array(self):
        cat = Categorical(['A', 'B', 'C', None, None])
        other = cat.fillna('C')
        result = cat.fillna(other)
        tm.assert_categorical_equal(result, other)
        assert isna(cat[-1])
        other = np.array(['A', 'B', 'C', 'B', 'A'])
        result = cat.fillna(other)
        expected = Categorical(['A', 'B', 'C', 'B', 'A'], dtype=cat.dtype)
        tm.assert_categorical_equal(result, expected)
        assert isna(cat[-1])

    @pytest.mark.parametrize('values, expected', [([1, 2, 3], np.array([False, False, False])), ([1, 2, np.nan], np.array([False, False, True])), ([1, 2, np.inf], np.array([False, False, True])), ([1, 2, pd.NA], np.array([False, False, True]))])
    def test_use_inf_as_na(self, values, expected):
        msg = 'use_inf_as_na option is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            with pd.option_context('mode.use_inf_as_na', True):
                cat = Categorical(values)
                result = cat.isna()
                tm.assert_numpy_array_equal(result, expected)
                result = Series(cat).isna()
                expected = Series(expected)
                tm.assert_series_equal(result, expected)
                result = DataFrame(cat).isna()
                expected = DataFrame(expected)
                tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('values, expected', [([1, 2, 3], np.array([False, False, False])), ([1, 2, np.nan], np.array([False, False, True])), ([1, 2, np.inf], np.array([False, False, True])), ([1, 2, pd.NA], np.array([False, False, True]))])
    def test_use_inf_as_na_outside_context(self, values, expected):
        cat = Categorical(values)
        msg = 'use_inf_as_na option is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            with pd.option_context('mode.use_inf_as_na', True):
                result = isna(cat)
                tm.assert_numpy_array_equal(result, expected)
                result = isna(Series(cat))
                expected = Series(expected)
                tm.assert_series_equal(result, expected)
                result = isna(DataFrame(cat))
                expected = DataFrame(expected)
                tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('a1, a2, categories', [(['a', 'b', 'c'], [np.nan, 'a', 'b'], ['a', 'b', 'c']), ([1, 2, 3], [np.nan, 1, 2], [1, 2, 3])])
    def test_compare_categorical_with_missing(self, a1, a2, categories):
        cat_type = CategoricalDtype(categories)
        result = Series(a1, dtype=cat_type) != Series(a2, dtype=cat_type)
        expected = Series(a1) != Series(a2)
        tm.assert_series_equal(result, expected)
        result = Series(a1, dtype=cat_type) == Series(a2, dtype=cat_type)
        expected = Series(a1) == Series(a2)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('na_value, dtype', [(pd.NaT, 'datetime64[ns]'), (None, 'float64'), (np.nan, 'float64'), (pd.NA, 'float64')])
    def test_categorical_only_missing_values_no_cast(self, na_value, dtype):
        result = Categorical([na_value, na_value])
        tm.assert_index_equal(result.categories, Index([], dtype=dtype))