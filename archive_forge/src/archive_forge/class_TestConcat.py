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
class TestConcat:
    data = {'a': [1, 2, 3], 'b': [10, 20, 30], 'd': [1000, 2000, 3000], 'e': [11, 22, 33]}
    data2 = {'a': [4, 5, 6], 'c': [400, 500, 600], 'b': [40, 50, 60], 'f': [444, 555, 666]}
    data3 = {'f': [2, 3, 4], 'g': [400, 500, 600], 'h': [20, 30, 40]}

    @pytest.mark.parametrize('join', ['inner', 'outer'])
    @pytest.mark.parametrize('sort', bool_arg_values)
    @pytest.mark.parametrize('ignore_index', bool_arg_values)
    def test_concat(self, join, sort, ignore_index):

        def concat(lib, df1, df2, join, sort, ignore_index):
            return lib.concat([df1, df2], join=join, sort=sort, ignore_index=ignore_index)
        run_and_compare(concat, data=self.data, data2=self.data2, join=join, sort=sort, ignore_index=ignore_index)

    def test_concat_with_same_df(self):

        def concat(df, **kwargs):
            df['f'] = df['a']
            return df
        run_and_compare(concat, data=self.data)

    def test_setitem_lazy(self):

        def applier(df, **kwargs):
            df = df + 1
            df['a'] = df['a'] + 1
            df['e'] = df['a'] + 1
            df['new_int8'] = np.int8(10)
            df['new_int16'] = np.int16(10)
            df['new_int32'] = np.int32(10)
            df['new_int64'] = np.int64(10)
            df['new_int'] = 10
            df['new_float'] = 5.5
            df['new_float64'] = np.float64(10.1)
            return df
        run_and_compare(applier, data=self.data)

    def test_setitem_default(self):

        def applier(df, lib, **kwargs):
            df = df + 1
            df['a'] = np.arange(3)
            df['b'] = lib.Series(np.arange(3))
            return df
        run_and_compare(applier, data=self.data, force_lazy=False)

    def test_insert_lazy(self):

        def applier(df, **kwargs):
            df = df + 1
            df.insert(2, 'new_int', 10)
            df.insert(1, 'new_float', 5.5)
            df.insert(0, 'new_a', df['a'] + 1)
            return df
        run_and_compare(applier, data=self.data)

    def test_insert_default(self):

        def applier(df, lib, **kwargs):
            df = df + 1
            df.insert(1, 'new_range', np.arange(3))
            df.insert(1, 'new_series', lib.Series(np.arange(3)))
            return df
        run_and_compare(applier, data=self.data, force_lazy=False)

    @pytest.mark.parametrize('data', [None, {'A': range(10)}, pandas.DataFrame({'A': range(10)})])
    @pytest.mark.parametrize('index', [None, pandas.RangeIndex(10), pandas.RangeIndex(start=10, stop=0, step=-1)])
    @pytest.mark.parametrize('value', [list(range(10)), pandas.Series(range(10))])
    @pytest.mark.parametrize('part_type', [None, 'arrow', 'hdk'])
    @pytest.mark.parametrize('insert_scalar', [True, False])
    def test_insert_list(self, data, index, value, part_type, insert_scalar):

        def create():
            mdf, pdf = create_test_dfs(data, index=index)
            if part_type == 'arrow':
                mdf._query_compiler._modin_frame._partitions[0][0].get(True)
            elif part_type == 'hdk':
                mdf._query_compiler._modin_frame.force_import()
            return (mdf, pdf)

        def insert(loc, name, value):
            nonlocal mdf, pdf
            mdf.insert(loc, name, value)
            pdf.insert(loc, name, value)
            if insert_scalar:
                mdf[f'S{loc}'] = 1
                pdf[f'S{loc}'] = 1
        niter = 3
        mdf, pdf = create()
        for i in range(niter):
            insert(len(pdf.columns), f'B{i}', value)
        df_equals(mdf, pdf)
        mdf, pdf = create()
        for i in range(niter):
            insert(0, f'C{i}', value)
        df_equals(mdf, pdf)
        mdf, pdf = create()
        for i in range(niter):
            insert(len(pdf.columns), f'B{i}', value)
            insert(0, f'C{i}', value)
            insert(len(pdf.columns) // 2, f'D{i}', value)
        df_equals(mdf, pdf)

    def test_concat_many(self):

        def concat(df1, df2, lib, **kwargs):
            df3 = df1.copy()
            df4 = df2.copy()
            return lib.concat([df1, df2, df3, df4])

        def sort_comparator(df1, df2):
            """Sort and verify equality of the passed frames."""
            df1, df2 = (try_cast_to_pandas(df).sort_values(df.columns[0]) for df in (df1, df2))
            return df_equals(df1, df2)
        run_and_compare(concat, data=self.data, data2=self.data2, comparator=sort_comparator, allow_subqueries=True)

    def test_concat_agg(self):

        def concat(lib, df1, df2):
            df1 = df1.groupby('a', as_index=False).agg({'b': 'sum', 'd': 'sum', 'e': 'sum'})
            df2 = df2.groupby('a', as_index=False).agg({'c': 'sum', 'b': 'sum', 'f': 'sum'})
            return lib.concat([df1, df2])
        run_and_compare(concat, data=self.data, data2=self.data2, allow_subqueries=True)

    @pytest.mark.parametrize('join', ['inner', 'outer'])
    @pytest.mark.parametrize('sort', bool_arg_values)
    @pytest.mark.parametrize('ignore_index', bool_arg_values)
    def test_concat_single(self, join, sort, ignore_index):

        def concat(lib, df, join, sort, ignore_index):
            return lib.concat([df], join=join, sort=sort, ignore_index=ignore_index)
        run_and_compare(concat, data=self.data, join=join, sort=sort, ignore_index=ignore_index)

    def test_groupby_concat_single(self):

        def concat(lib, df):
            df = lib.concat([df])
            return df.groupby('a').agg({'b': 'min'})
        run_and_compare(concat, data=self.data)

    @pytest.mark.parametrize('join', ['inner'])
    @pytest.mark.parametrize('sort', bool_arg_values)
    @pytest.mark.parametrize('ignore_index', bool_arg_values)
    def test_concat_join(self, join, sort, ignore_index):

        def concat(lib, df1, df2, join, sort, ignore_index, **kwargs):
            return lib.concat([df1, df2], axis=1, join=join, sort=sort, ignore_index=ignore_index)
        run_and_compare(concat, data=self.data, data2=self.data3, join=join, sort=sort, ignore_index=ignore_index)

    def test_concat_index_name(self):
        df1 = pandas.DataFrame(self.data)
        df1 = df1.set_index('a')
        df2 = pandas.DataFrame(self.data3)
        df2 = df2.set_index('f')
        ref = pandas.concat([df1, df2], axis=1, join='inner')
        exp = pd.concat([df1, df2], axis=1, join='inner')
        df_equals(ref, exp)
        df2.index.name = 'a'
        ref = pandas.concat([df1, df2], axis=1, join='inner')
        exp = pd.concat([df1, df2], axis=1, join='inner')
        df_equals(ref, exp)

    def test_concat_index_names(self):
        df1 = pandas.DataFrame(self.data)
        df1 = df1.set_index(['a', 'b'])
        df2 = pandas.DataFrame(self.data3)
        df2 = df2.set_index(['f', 'h'])
        ref = pandas.concat([df1, df2], axis=1, join='inner')
        exp = pd.concat([df1, df2], axis=1, join='inner')
        df_equals(ref, exp)
        df2.index.names = ['a', 'b']
        ref = pandas.concat([df1, df2], axis=1, join='inner')
        exp = pd.concat([df1, df2], axis=1, join='inner')
        df_equals(ref, exp)

    def test_concat_str(self):

        def concat(df1, df2, lib, **kwargs):
            return lib.concat([df1.dropna(), df2.dropna()]).astype(str)
        run_and_compare(concat, data={'a': ['1', '2', '3']}, data2={'a': ['4', '5', '6']}, force_lazy=False)

    @pytest.mark.parametrize('transform', [True, False])
    @pytest.mark.parametrize('sort_last', [True, False])
    def test_issue_5889(self, transform, sort_last):
        with ensure_clean('.csv') as file:
            data = {'a': [1, 2, 3], 'b': [1, 2, 3]} if transform else {'a': [1, 2, 3]}
            pandas.DataFrame(data).to_csv(file, index=False)

            def test_concat(lib, **kwargs):
                if transform:

                    def read_csv():
                        return lib.read_csv(file)['b']
                else:

                    def read_csv():
                        return lib.read_csv(file)
                df = read_csv()
                for _ in range(100):
                    df = lib.concat([df, read_csv()])
                if sort_last:
                    df = lib.concat([df, read_csv()], sort=True)
                return df
            run_and_compare(test_concat, data={})