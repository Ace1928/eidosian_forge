import os
import re
import numpy as np
import pandas
import pyarrow
import pytest
from pandas._testing import ensure_clean
from pandas.core.dtypes.common import is_list_like
from pyhdk import __version__ as hdk_version
from modin.config import StorageFormat
from modin.tests.interchange.dataframe_protocol.hdk.utils import split_df_into_chunks
from modin.tests.pandas.utils import (
from .utils import ForceHdkImport, eval_io, run_and_compare, set_execution_mode
import modin.pandas as pd
from modin.experimental.core.execution.native.implementations.hdk_on_native.calcite_serializer import (
from modin.experimental.core.execution.native.implementations.hdk_on_native.df_algebra import (
from modin.experimental.core.execution.native.implementations.hdk_on_native.partitioning.partition_manager import (
from modin.pandas.io import from_arrow
from modin.tests.pandas.utils import (
from modin.utils import try_cast_to_pandas
class TestSort:
    data = {'a': [1, 2, 5, -2, -5, 4, -4, 6, 3], 'b': [1, 2, 3, 6, 5, 1, 4, 5, 3], 'c': [5, 4, 2, 3, 1, 1, 4, 5, 6], 'd': ['1', '4', '3', '2', '1', '6', '7', '5', '0']}
    data_nulls = {'a': [1, 2, 5, -2, -5, 4, -4, None, 3], 'b': [1, 2, 3, 6, 5, None, 4, 5, 3], 'c': [None, 4, 2, 3, 1, 1, 4, 5, 6]}
    data_multiple_nulls = {'a': [1, 2, None, -2, 5, 4, -4, None, 3], 'b': [1, 2, 3, 6, 5, None, 4, 5, None], 'c': [None, 4, 2, None, 1, 1, 4, 5, 6]}
    cols_values = ['a', ['a', 'b'], ['b', 'a'], ['c', 'a', 'b']]
    index_cols_values = [None, 'a', ['a', 'b']]
    ascending_values = [True, False]
    ascending_list_values = [[True, False], [False, True]]
    na_position_values = ['first', 'last']

    @pytest.mark.parametrize('cols', cols_values)
    @pytest.mark.parametrize('ignore_index', bool_arg_values)
    @pytest.mark.parametrize('ascending', ascending_values)
    @pytest.mark.parametrize('index_cols', index_cols_values)
    def test_sort_cols(self, cols, ignore_index, index_cols, ascending):

        def sort(df, cols, ignore_index, index_cols, ascending, **kwargs):
            if index_cols:
                df = df.set_index(index_cols)
                df_equals_with_non_stable_indices()
            return df.sort_values(cols, ignore_index=ignore_index, ascending=ascending)
        run_and_compare(sort, data=self.data, cols=cols, ignore_index=ignore_index, index_cols=index_cols, ascending=ascending, force_lazy=index_cols is None)

    @pytest.mark.parametrize('ascending', ascending_list_values)
    def test_sort_cols_asc_list(self, ascending):

        def sort(df, ascending, **kwargs):
            return df.sort_values(['a', 'b'], ascending=ascending)
        run_and_compare(sort, data=self.data, ascending=ascending)

    @pytest.mark.skipif(hdk_version == '0.7.0', reason='https://github.com/modin-project/modin/issues/6514')
    @pytest.mark.parametrize('ascending', ascending_values)
    def test_sort_cols_str(self, ascending):

        def sort(df, ascending, **kwargs):
            return df.sort_values('d', ascending=ascending)
        run_and_compare(sort, data=self.data, ascending=ascending)

    @pytest.mark.parametrize('cols', cols_values)
    @pytest.mark.parametrize('ascending', ascending_values)
    @pytest.mark.parametrize('na_position', na_position_values)
    def test_sort_cols_nulls(self, cols, ascending, na_position):

        def sort(df, cols, ascending, na_position, **kwargs):
            return df.sort_values(cols, ascending=ascending, na_position=na_position)
        run_and_compare(sort, data=self.data_nulls, cols=cols, ascending=ascending, na_position=na_position)