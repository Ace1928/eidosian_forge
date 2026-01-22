from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
class TestSetOpsUnsorted:

    def test_intersect_str_dates(self):
        dt_dates = [datetime(2012, 2, 9), datetime(2012, 2, 22)]
        index1 = Index(dt_dates, dtype=object)
        index2 = Index(['aa'], dtype=object)
        result = index2.intersection(index1)
        expected = Index([], dtype=object)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('index', ['string'], indirect=True)
    def test_intersection(self, index, sort):
        first = index[:20]
        second = index[:10]
        intersect = first.intersection(second, sort=sort)
        if sort in (None, False):
            tm.assert_index_equal(intersect.sort_values(), second.sort_values())
        else:
            tm.assert_index_equal(intersect, second)
        inter = first.intersection(first, sort=sort)
        assert inter is first

    @pytest.mark.parametrize('index2,keeps_name', [(Index([3, 4, 5, 6, 7], name='index'), True), (Index([3, 4, 5, 6, 7], name='other'), False), (Index([3, 4, 5, 6, 7]), False)])
    def test_intersection_name_preservation(self, index2, keeps_name, sort):
        index1 = Index([1, 2, 3, 4, 5], name='index')
        expected = Index([3, 4, 5])
        result = index1.intersection(index2, sort)
        if keeps_name:
            expected.name = 'index'
        assert result.name == expected.name
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('index', ['string'], indirect=True)
    @pytest.mark.parametrize('first_name,second_name,expected_name', [('A', 'A', 'A'), ('A', 'B', None), (None, 'B', None)])
    def test_intersection_name_preservation2(self, index, first_name, second_name, expected_name, sort):
        first = index[5:20]
        second = index[:10]
        first.name = first_name
        second.name = second_name
        intersect = first.intersection(second, sort=sort)
        assert intersect.name == expected_name

    def test_chained_union(self, sort):
        i1 = Index([1, 2], name='i1')
        i2 = Index([5, 6], name='i2')
        i3 = Index([3, 4], name='i3')
        union = i1.union(i2.union(i3, sort=sort), sort=sort)
        expected = i1.union(i2, sort=sort).union(i3, sort=sort)
        tm.assert_index_equal(union, expected)
        j1 = Index([1, 2], name='j1')
        j2 = Index([], name='j2')
        j3 = Index([], name='j3')
        union = j1.union(j2.union(j3, sort=sort), sort=sort)
        expected = j1.union(j2, sort=sort).union(j3, sort=sort)
        tm.assert_index_equal(union, expected)

    @pytest.mark.parametrize('index', ['string'], indirect=True)
    def test_union(self, index, sort):
        first = index[5:20]
        second = index[:10]
        everything = index[:20]
        union = first.union(second, sort=sort)
        if sort in (None, False):
            tm.assert_index_equal(union.sort_values(), everything.sort_values())
        else:
            tm.assert_index_equal(union, everything)

    @pytest.mark.parametrize('klass', [np.array, Series, list])
    @pytest.mark.parametrize('index', ['string'], indirect=True)
    def test_union_from_iterables(self, index, klass, sort):
        first = index[5:20]
        second = index[:10]
        everything = index[:20]
        case = klass(second.values)
        result = first.union(case, sort=sort)
        if sort in (None, False):
            tm.assert_index_equal(result.sort_values(), everything.sort_values())
        else:
            tm.assert_index_equal(result, everything)

    @pytest.mark.parametrize('index', ['string'], indirect=True)
    def test_union_identity(self, index, sort):
        first = index[5:20]
        union = first.union(first, sort=sort)
        assert (union is first) is (not sort)
        union = first.union(Index([], dtype=first.dtype), sort=sort)
        assert (union is first) is (not sort)
        union = Index([], dtype=first.dtype).union(first, sort=sort)
        assert (union is first) is (not sort)

    @pytest.mark.parametrize('index', ['string'], indirect=True)
    @pytest.mark.parametrize('second_name,expected', [(None, None), ('name', 'name')])
    def test_difference_name_preservation(self, index, second_name, expected, sort):
        first = index[5:20]
        second = index[:10]
        answer = index[10:20]
        first.name = 'name'
        second.name = second_name
        result = first.difference(second, sort=sort)
        if sort is True:
            tm.assert_index_equal(result, answer)
        else:
            answer.name = second_name
            tm.assert_index_equal(result.sort_values(), answer.sort_values())
        if expected is None:
            assert result.name is None
        else:
            assert result.name == expected

    def test_difference_empty_arg(self, index, sort):
        first = index.copy()
        first = first[5:20]
        first.name = 'name'
        result = first.difference([], sort)
        expected = index[5:20].unique()
        expected.name = 'name'
        tm.assert_index_equal(result, expected)

    def test_difference_should_not_compare(self):
        left = Index([1, 1])
        right = Index([True])
        result = left.difference(right)
        expected = Index([1])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('index', ['string'], indirect=True)
    def test_difference_identity(self, index, sort):
        first = index[5:20]
        first.name = 'name'
        result = first.difference(first, sort)
        assert len(result) == 0
        assert result.name == first.name

    @pytest.mark.parametrize('index', ['string'], indirect=True)
    def test_difference_sort(self, index, sort):
        first = index[5:20]
        second = index[:10]
        result = first.difference(second, sort)
        expected = index[10:20]
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('opname', ['difference', 'symmetric_difference'])
    def test_difference_incomparable(self, opname):
        a = Index([3, Timestamp('2000'), 1])
        b = Index([2, Timestamp('1999'), 1])
        op = operator.methodcaller(opname, b)
        with tm.assert_produces_warning(RuntimeWarning):
            result = op(a)
        expected = Index([3, Timestamp('2000'), 2, Timestamp('1999')])
        if opname == 'difference':
            expected = expected[:2]
        tm.assert_index_equal(result, expected)
        op = operator.methodcaller(opname, b, sort=False)
        result = op(a)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('opname', ['difference', 'symmetric_difference'])
    def test_difference_incomparable_true(self, opname):
        a = Index([3, Timestamp('2000'), 1])
        b = Index([2, Timestamp('1999'), 1])
        op = operator.methodcaller(opname, b, sort=True)
        msg = "'<' not supported between instances of 'Timestamp' and 'int'"
        with pytest.raises(TypeError, match=msg):
            op(a)

    def test_symmetric_difference_mi(self, sort):
        index1 = MultiIndex.from_tuples(zip(['foo', 'bar', 'baz'], [1, 2, 3]))
        index2 = MultiIndex.from_tuples([('foo', 1), ('bar', 3)])
        result = index1.symmetric_difference(index2, sort=sort)
        expected = MultiIndex.from_tuples([('bar', 2), ('baz', 3), ('bar', 3)])
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('index2,expected', [(Index([0, 1, np.nan]), Index([2.0, 3.0, 0.0])), (Index([0, 1]), Index([np.nan, 2.0, 3.0, 0.0]))])
    def test_symmetric_difference_missing(self, index2, expected, sort):
        index1 = Index([1, np.nan, 2, 3])
        result = index1.symmetric_difference(index2, sort=sort)
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)

    def test_symmetric_difference_non_index(self, sort):
        index1 = Index([1, 2, 3, 4], name='index1')
        index2 = np.array([2, 3, 4, 5])
        expected = Index([1, 5], name='index1')
        result = index1.symmetric_difference(index2, sort=sort)
        if sort in (None, True):
            tm.assert_index_equal(result, expected)
        else:
            tm.assert_index_equal(result.sort_values(), expected)
        assert result.name == 'index1'
        result = index1.symmetric_difference(index2, result_name='new_name', sort=sort)
        expected.name = 'new_name'
        if sort in (None, True):
            tm.assert_index_equal(result, expected)
        else:
            tm.assert_index_equal(result.sort_values(), expected)
        assert result.name == 'new_name'

    def test_union_ea_dtypes(self, any_numeric_ea_and_arrow_dtype):
        idx = Index([1, 2, 3], dtype=any_numeric_ea_and_arrow_dtype)
        idx2 = Index([3, 4, 5], dtype=any_numeric_ea_and_arrow_dtype)
        result = idx.union(idx2)
        expected = Index([1, 2, 3, 4, 5], dtype=any_numeric_ea_and_arrow_dtype)
        tm.assert_index_equal(result, expected)

    def test_union_string_array(self, any_string_dtype):
        idx1 = Index(['a'], dtype=any_string_dtype)
        idx2 = Index(['b'], dtype=any_string_dtype)
        result = idx1.union(idx2)
        expected = Index(['a', 'b'], dtype=any_string_dtype)
        tm.assert_index_equal(result, expected)