import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
class TestJSONArray(base.ExtensionTests):

    @pytest.mark.xfail(reason='comparison method not implemented for JSONArray (GH-37867)')
    def test_contains(self, data):
        super().test_contains(data)

    @pytest.mark.xfail(reason='not implemented constructor from dtype')
    def test_from_dtype(self, data):
        super().test_from_dtype(data)

    @pytest.mark.xfail(reason='RecursionError, GH-33900')
    def test_series_constructor_no_data_with_index(self, dtype, na_value):
        rec_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(100)
            super().test_series_constructor_no_data_with_index(dtype, na_value)
        finally:
            sys.setrecursionlimit(rec_limit)

    @pytest.mark.xfail(reason='RecursionError, GH-33900')
    def test_series_constructor_scalar_na_with_index(self, dtype, na_value):
        rec_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(100)
            super().test_series_constructor_scalar_na_with_index(dtype, na_value)
        finally:
            sys.setrecursionlimit(rec_limit)

    @pytest.mark.xfail(reason='collection as scalar, GH-33901')
    def test_series_constructor_scalar_with_index(self, data, dtype):
        rec_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(100)
            super().test_series_constructor_scalar_with_index(data, dtype)
        finally:
            sys.setrecursionlimit(rec_limit)

    @pytest.mark.xfail(reason='Different definitions of NA')
    def test_stack(self):
        """
        The test does .astype(object).stack(future_stack=True). If we happen to have
        any missing values in `data`, then we'll end up with different
        rows since we consider `{}` NA, but `.astype(object)` doesn't.
        """
        super().test_stack()

    @pytest.mark.xfail(reason='dict for NA')
    def test_unstack(self, data, index):
        return super().test_unstack(data, index)

    @pytest.mark.xfail(reason='Setting a dict as a scalar')
    def test_fillna_series(self):
        """We treat dictionaries as a mapping in fillna, not a scalar."""
        super().test_fillna_series()

    @pytest.mark.xfail(reason='Setting a dict as a scalar')
    def test_fillna_frame(self):
        """We treat dictionaries as a mapping in fillna, not a scalar."""
        super().test_fillna_frame()

    @pytest.mark.parametrize('limit_area, input_ilocs, expected_ilocs', [('outside', [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]), ('outside', [1, 0, 1, 0, 1], [1, 0, 1, 0, 1]), ('outside', [0, 1, 1, 1, 0], [0, 1, 1, 1, 1]), ('outside', [0, 1, 0, 1, 0], [0, 1, 0, 1, 1]), ('inside', [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]), ('inside', [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]), ('inside', [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]), ('inside', [0, 1, 0, 1, 0], [0, 1, 1, 1, 0])])
    def test_ffill_limit_area(self, data_missing, limit_area, input_ilocs, expected_ilocs):
        msg = 'JSONArray does not implement limit_area'
        with pytest.raises(NotImplementedError, match=msg):
            super().test_ffill_limit_area(data_missing, limit_area, input_ilocs, expected_ilocs)

    @unhashable
    def test_value_counts(self, all_data, dropna):
        super().test_value_counts(all_data, dropna)

    @unhashable
    def test_value_counts_with_normalize(self, data):
        super().test_value_counts_with_normalize(data)

    @unhashable
    def test_sort_values_frame(self):
        super().test_sort_values_frame()

    @pytest.mark.parametrize('ascending', [True, False])
    def test_sort_values(self, data_for_sorting, ascending, sort_by_key):
        super().test_sort_values(data_for_sorting, ascending, sort_by_key)

    @pytest.mark.parametrize('ascending', [True, False])
    def test_sort_values_missing(self, data_missing_for_sorting, ascending, sort_by_key):
        super().test_sort_values_missing(data_missing_for_sorting, ascending, sort_by_key)

    @pytest.mark.xfail(reason='combine for JSONArray not supported')
    def test_combine_le(self, data_repeated):
        super().test_combine_le(data_repeated)

    @pytest.mark.xfail(reason='combine for JSONArray not supported - may pass depending on random data', strict=False, raises=AssertionError)
    def test_combine_first(self, data):
        super().test_combine_first(data)

    @pytest.mark.xfail(reason='broadcasting error')
    def test_where_series(self, data, na_value):
        super().test_where_series(data, na_value)

    @pytest.mark.xfail(reason="Can't compare dicts.")
    def test_searchsorted(self, data_for_sorting):
        super().test_searchsorted(data_for_sorting)

    @pytest.mark.xfail(reason="Can't compare dicts.")
    def test_equals(self, data, na_value, as_series):
        super().test_equals(data, na_value, as_series)

    @pytest.mark.skip('fill-value is interpreted as a dict of values')
    def test_fillna_copy_frame(self, data_missing):
        super().test_fillna_copy_frame(data_missing)

    def test_equals_same_data_different_object(self, data, using_copy_on_write, request):
        if using_copy_on_write:
            mark = pytest.mark.xfail(reason='Fails with CoW')
            request.applymarker(mark)
        super().test_equals_same_data_different_object(data)

    @pytest.mark.xfail(reason='failing on np.array(self, dtype=str)')
    def test_astype_str(self):
        """This currently fails in NumPy on np.array(self, dtype=str) with

        *** ValueError: setting an array element with a sequence
        """
        super().test_astype_str()

    @unhashable
    def test_groupby_extension_transform(self):
        """
        This currently fails in Series.name.setter, since the
        name must be hashable, but the value is a dictionary.
        I think this is what we want, i.e. `.name` should be the original
        values, and not the values for factorization.
        """
        super().test_groupby_extension_transform()

    @unhashable
    def test_groupby_extension_apply(self):
        """
        This fails in Index._do_unique_check with

        >   hash(val)
        E   TypeError: unhashable type: 'UserDict' with

        I suspect that once we support Index[ExtensionArray],
        we'll be able to dispatch unique.
        """
        super().test_groupby_extension_apply()

    @unhashable
    def test_groupby_extension_agg(self):
        """
        This fails when we get to tm.assert_series_equal when left.index
        contains dictionaries, which are not hashable.
        """
        super().test_groupby_extension_agg()

    @unhashable
    def test_groupby_extension_no_sort(self):
        """
        This fails when we get to tm.assert_series_equal when left.index
        contains dictionaries, which are not hashable.
        """
        super().test_groupby_extension_no_sort()

    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators, request):
        if len(data[0]) != 1:
            mark = pytest.mark.xfail(reason='raises in coercing to Series')
            request.applymarker(mark)
        super().test_arith_frame_with_scalar(data, all_arithmetic_operators)

    def test_compare_array(self, data, comparison_op, request):
        if comparison_op.__name__ in ['eq', 'ne']:
            mark = pytest.mark.xfail(reason='Comparison methods not implemented')
            request.applymarker(mark)
        super().test_compare_array(data, comparison_op)

    @pytest.mark.xfail(reason='ValueError: Must have equal len keys and value')
    def test_setitem_loc_scalar_mixed(self, data):
        super().test_setitem_loc_scalar_mixed(data)

    @pytest.mark.xfail(reason='ValueError: Must have equal len keys and value')
    def test_setitem_loc_scalar_multiple_homogoneous(self, data):
        super().test_setitem_loc_scalar_multiple_homogoneous(data)

    @pytest.mark.xfail(reason='ValueError: Must have equal len keys and value')
    def test_setitem_iloc_scalar_mixed(self, data):
        super().test_setitem_iloc_scalar_mixed(data)

    @pytest.mark.xfail(reason='ValueError: Must have equal len keys and value')
    def test_setitem_iloc_scalar_multiple_homogoneous(self, data):
        super().test_setitem_iloc_scalar_multiple_homogoneous(data)

    @pytest.mark.parametrize('mask', [np.array([True, True, True, False, False]), pd.array([True, True, True, False, False], dtype='boolean'), pd.array([True, True, True, pd.NA, pd.NA], dtype='boolean')], ids=['numpy-array', 'boolean-array', 'boolean-array-na'])
    def test_setitem_mask(self, data, mask, box_in_series, request):
        if box_in_series:
            mark = pytest.mark.xfail(reason='cannot set using a list-like indexer with a different length')
            request.applymarker(mark)
        elif not isinstance(mask, np.ndarray):
            mark = pytest.mark.xfail(reason='Issues unwanted DeprecationWarning')
            request.applymarker(mark)
        super().test_setitem_mask(data, mask, box_in_series)

    def test_setitem_mask_raises(self, data, box_in_series, request):
        if not box_in_series:
            mark = pytest.mark.xfail(reason='Fails to raise')
            request.applymarker(mark)
        super().test_setitem_mask_raises(data, box_in_series)

    @pytest.mark.xfail(reason='cannot set using a list-like indexer with a different length')
    def test_setitem_mask_boolean_array_with_na(self, data, box_in_series):
        super().test_setitem_mask_boolean_array_with_na(data, box_in_series)

    @pytest.mark.parametrize('idx', [[0, 1, 2], pd.array([0, 1, 2], dtype='Int64'), np.array([0, 1, 2])], ids=['list', 'integer-array', 'numpy-array'])
    def test_setitem_integer_array(self, data, idx, box_in_series, request):
        if box_in_series:
            mark = pytest.mark.xfail(reason='cannot set using a list-like indexer with a different length')
            request.applymarker(mark)
        super().test_setitem_integer_array(data, idx, box_in_series)

    @pytest.mark.xfail(reason='list indices must be integers or slices, not NAType')
    @pytest.mark.parametrize('idx, box_in_series', [([0, 1, 2, pd.NA], False), pytest.param([0, 1, 2, pd.NA], True, marks=pytest.mark.xfail(reason='GH-31948')), (pd.array([0, 1, 2, pd.NA], dtype='Int64'), False), (pd.array([0, 1, 2, pd.NA], dtype='Int64'), False)], ids=['list-False', 'list-True', 'integer-array-False', 'integer-array-True'])
    def test_setitem_integer_with_missing_raises(self, data, idx, box_in_series):
        super().test_setitem_integer_with_missing_raises(data, idx, box_in_series)

    @pytest.mark.xfail(reason='Fails to raise')
    def test_setitem_scalar_key_sequence_raise(self, data):
        super().test_setitem_scalar_key_sequence_raise(data)

    def test_setitem_with_expansion_dataframe_column(self, data, full_indexer, request):
        if 'full_slice' in request.node.name:
            mark = pytest.mark.xfail(reason='slice is not iterable')
            request.applymarker(mark)
        super().test_setitem_with_expansion_dataframe_column(data, full_indexer)

    @pytest.mark.xfail(reason='slice is not iterable')
    def test_setitem_frame_2d_values(self, data):
        super().test_setitem_frame_2d_values(data)

    @pytest.mark.xfail(reason='cannot set using a list-like indexer with a different length')
    @pytest.mark.parametrize('setter', ['loc', None])
    def test_setitem_mask_broadcast(self, data, setter):
        super().test_setitem_mask_broadcast(data, setter)

    @pytest.mark.xfail(reason='cannot set using a slice indexer with a different length')
    def test_setitem_slice(self, data, box_in_series):
        super().test_setitem_slice(data, box_in_series)

    @pytest.mark.xfail(reason='slice object is not iterable')
    def test_setitem_loc_iloc_slice(self, data):
        super().test_setitem_loc_iloc_slice(data)

    @pytest.mark.xfail(reason='slice object is not iterable')
    def test_setitem_slice_mismatch_length_raises(self, data):
        super().test_setitem_slice_mismatch_length_raises(data)

    @pytest.mark.xfail(reason='slice object is not iterable')
    def test_setitem_slice_array(self, data):
        super().test_setitem_slice_array(data)

    @pytest.mark.xfail(reason='Fail to raise')
    def test_setitem_invalid(self, data, invalid_scalar):
        super().test_setitem_invalid(data, invalid_scalar)

    @pytest.mark.xfail(reason='only integer scalar arrays can be converted')
    def test_setitem_2d_values(self, data):
        super().test_setitem_2d_values(data)

    @pytest.mark.xfail(reason="data type 'json' not understood")
    @pytest.mark.parametrize('engine', ['c', 'python'])
    def test_EA_types(self, engine, data, request):
        super().test_EA_types(engine, data, request)