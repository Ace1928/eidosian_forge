import re
import weakref
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
class TestCategoricalDtypeParametrized:

    @pytest.mark.parametrize('categories', [list('abcd'), np.arange(1000), ['a', 'b', 10, 2, 1.3, True], [True, False], date_range('2017', periods=4)])
    def test_basic(self, categories, ordered):
        c1 = CategoricalDtype(categories, ordered=ordered)
        tm.assert_index_equal(c1.categories, pd.Index(categories))
        assert c1.ordered is ordered

    def test_order_matters(self):
        categories = ['a', 'b']
        c1 = CategoricalDtype(categories, ordered=True)
        c2 = CategoricalDtype(categories, ordered=False)
        c3 = CategoricalDtype(categories, ordered=None)
        assert c1 is not c2
        assert c1 is not c3

    @pytest.mark.parametrize('ordered', [False, None])
    def test_unordered_same(self, ordered):
        c1 = CategoricalDtype(['a', 'b'], ordered=ordered)
        c2 = CategoricalDtype(['b', 'a'], ordered=ordered)
        assert hash(c1) == hash(c2)

    def test_categories(self):
        result = CategoricalDtype(['a', 'b', 'c'])
        tm.assert_index_equal(result.categories, pd.Index(['a', 'b', 'c']))
        assert result.ordered is False

    def test_equal_but_different(self):
        c1 = CategoricalDtype([1, 2, 3])
        c2 = CategoricalDtype([1.0, 2.0, 3.0])
        assert c1 is not c2
        assert c1 != c2

    def test_equal_but_different_mixed_dtypes(self):
        c1 = CategoricalDtype([1, 2, '3'])
        c2 = CategoricalDtype(['3', 1, 2])
        assert c1 is not c2
        assert c1 == c2

    def test_equal_empty_ordered(self):
        c1 = CategoricalDtype([], ordered=True)
        c2 = CategoricalDtype([], ordered=True)
        assert c1 is not c2
        assert c1 == c2

    def test_equal_empty_unordered(self):
        c1 = CategoricalDtype([])
        c2 = CategoricalDtype([])
        assert c1 is not c2
        assert c1 == c2

    @pytest.mark.parametrize('v1, v2', [([1, 2, 3], [1, 2, 3]), ([1, 2, 3], [3, 2, 1])])
    def test_order_hashes_different(self, v1, v2):
        c1 = CategoricalDtype(v1, ordered=False)
        c2 = CategoricalDtype(v2, ordered=True)
        c3 = CategoricalDtype(v1, ordered=None)
        assert c1 is not c2
        assert c1 is not c3

    def test_nan_invalid(self):
        msg = 'Categorical categories cannot be null'
        with pytest.raises(ValueError, match=msg):
            CategoricalDtype([1, 2, np.nan])

    def test_non_unique_invalid(self):
        msg = 'Categorical categories must be unique'
        with pytest.raises(ValueError, match=msg):
            CategoricalDtype([1, 2, 1])

    def test_same_categories_different_order(self):
        c1 = CategoricalDtype(['a', 'b'], ordered=True)
        c2 = CategoricalDtype(['b', 'a'], ordered=True)
        assert c1 is not c2

    @pytest.mark.parametrize('ordered1', [True, False, None])
    @pytest.mark.parametrize('ordered2', [True, False, None])
    def test_categorical_equality(self, ordered1, ordered2):
        c1 = CategoricalDtype(list('abc'), ordered1)
        c2 = CategoricalDtype(list('abc'), ordered2)
        result = c1 == c2
        expected = bool(ordered1) is bool(ordered2)
        assert result is expected
        c1 = CategoricalDtype(list('abc'), ordered1)
        c2 = CategoricalDtype(list('cab'), ordered2)
        result = c1 == c2
        expected = bool(ordered1) is False and bool(ordered2) is False
        assert result is expected
        c2 = CategoricalDtype([1, 2, 3], ordered2)
        assert c1 != c2
        c1 = CategoricalDtype(list('abc'), ordered1)
        c2 = CategoricalDtype(None, ordered2)
        c3 = CategoricalDtype(None, ordered1)
        assert c1 != c2
        assert c2 != c1
        assert c2 == c3

    def test_categorical_dtype_equality_requires_categories(self):
        first = CategoricalDtype(['a', 'b'])
        second = CategoricalDtype()
        third = CategoricalDtype(ordered=True)
        assert second == second
        assert third == third
        assert first != second
        assert second != first
        assert first != third
        assert third != first
        assert second == third
        assert third == second

    @pytest.mark.parametrize('categories', [list('abc'), None])
    @pytest.mark.parametrize('other', ['category', 'not a category'])
    def test_categorical_equality_strings(self, categories, ordered, other):
        c1 = CategoricalDtype(categories, ordered)
        result = c1 == other
        expected = other == 'category'
        assert result is expected

    def test_invalid_raises(self):
        with pytest.raises(TypeError, match='ordered'):
            CategoricalDtype(['a', 'b'], ordered='foo')
        with pytest.raises(TypeError, match="'categories' must be list-like"):
            CategoricalDtype('category')

    def test_mixed(self):
        a = CategoricalDtype(['a', 'b', 1, 2])
        b = CategoricalDtype(['a', 'b', '1', '2'])
        assert hash(a) != hash(b)

    def test_from_categorical_dtype_identity(self):
        c1 = Categorical([1, 2], categories=[1, 2, 3], ordered=True)
        c2 = CategoricalDtype._from_categorical_dtype(c1)
        assert c2 is c1

    def test_from_categorical_dtype_categories(self):
        c1 = Categorical([1, 2], categories=[1, 2, 3], ordered=True)
        result = CategoricalDtype._from_categorical_dtype(c1, categories=[2, 3])
        assert result == CategoricalDtype([2, 3], ordered=True)

    def test_from_categorical_dtype_ordered(self):
        c1 = Categorical([1, 2], categories=[1, 2, 3], ordered=True)
        result = CategoricalDtype._from_categorical_dtype(c1, ordered=False)
        assert result == CategoricalDtype([1, 2, 3], ordered=False)

    def test_from_categorical_dtype_both(self):
        c1 = Categorical([1, 2], categories=[1, 2, 3], ordered=True)
        result = CategoricalDtype._from_categorical_dtype(c1, categories=[1, 2], ordered=False)
        assert result == CategoricalDtype([1, 2], ordered=False)

    def test_str_vs_repr(self, ordered, using_infer_string):
        c1 = CategoricalDtype(['a', 'b'], ordered=ordered)
        assert str(c1) == 'category'
        dtype = 'string' if using_infer_string else 'object'
        pat = f'CategoricalDtype\\(categories=\\[.*\\], ordered={{ordered}}, categories_dtype={dtype}\\)'
        assert re.match(pat.format(ordered=ordered), repr(c1))

    def test_categorical_categories(self):
        c1 = CategoricalDtype(Categorical(['a', 'b']))
        tm.assert_index_equal(c1.categories, pd.Index(['a', 'b']))
        c1 = CategoricalDtype(CategoricalIndex(['a', 'b']))
        tm.assert_index_equal(c1.categories, pd.Index(['a', 'b']))

    @pytest.mark.parametrize('new_categories', [list('abc'), list('cba'), list('wxyz'), None])
    @pytest.mark.parametrize('new_ordered', [True, False, None])
    def test_update_dtype(self, ordered, new_categories, new_ordered):
        original_categories = list('abc')
        dtype = CategoricalDtype(original_categories, ordered)
        new_dtype = CategoricalDtype(new_categories, new_ordered)
        result = dtype.update_dtype(new_dtype)
        expected_categories = pd.Index(new_categories or original_categories)
        expected_ordered = new_ordered if new_ordered is not None else dtype.ordered
        tm.assert_index_equal(result.categories, expected_categories)
        assert result.ordered is expected_ordered

    def test_update_dtype_string(self, ordered):
        dtype = CategoricalDtype(list('abc'), ordered)
        expected_categories = dtype.categories
        expected_ordered = dtype.ordered
        result = dtype.update_dtype('category')
        tm.assert_index_equal(result.categories, expected_categories)
        assert result.ordered is expected_ordered

    @pytest.mark.parametrize('bad_dtype', ['foo', object, np.int64, PeriodDtype('Q')])
    def test_update_dtype_errors(self, bad_dtype):
        dtype = CategoricalDtype(list('abc'), False)
        msg = 'a CategoricalDtype must be passed to perform an update, '
        with pytest.raises(ValueError, match=msg):
            dtype.update_dtype(bad_dtype)