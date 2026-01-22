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
class TestBinaryOp:
    data = {'a': [1, 1, 1, 1, 1], 'b': [10, 10, 10, 10, 10], 'c': [100, 100, 100, 100, 100], 'd': [1000, 1000, 1000, 1000, 1000]}
    data2 = {'a': [1, 1, 1, 1, 1], 'f': [2, 2, 2, 2, 2], 'b': [3, 3, 3, 3, 3], 'd': [4, 4, 4, 4, 4]}
    fill_values = [None, 1]

    def test_binary_level(self):

        def applier(df1, df2, **kwargs):
            df2.index = generate_multiindex(len(df2))
            return df1.add(df2, level=1)
        run_and_compare(applier, data=self.data, data2=self.data, force_lazy=False)

    def test_add_cst(self):

        def add(df, **kwargs):
            return df + 1
        run_and_compare(add, data=self.data)

    def test_add_list(self):

        def add(df, **kwargs):
            return df + [1, 2, 3, 4]
        run_and_compare(add, data=self.data)

    @pytest.mark.parametrize('fill_value', fill_values)
    def test_add_method_columns(self, fill_value):

        def add1(df, fill_value, **kwargs):
            return df['a'].add(df['b'], fill_value=fill_value)

        def add2(df, fill_value, **kwargs):
            return df[['a', 'c']].add(df[['b', 'a']], fill_value=fill_value)
        run_and_compare(add1, data=self.data, fill_value=fill_value)
        run_and_compare(add2, data=self.data, fill_value=fill_value)

    def test_add_columns(self):

        def add1(df, **kwargs):
            return df['a'] + df['b']

        def add2(df, **kwargs):
            return df[['a', 'c']] + df[['b', 'a']]
        run_and_compare(add1, data=self.data)
        run_and_compare(add2, data=self.data)

    def test_add_columns_and_assign(self):

        def add(df, **kwargs):
            df['sum'] = df['a'] + df['b']
            return df
        run_and_compare(add, data=self.data)

    def test_add_columns_and_assign_to_existing(self):

        def add(df, **kwargs):
            df['a'] = df['a'] + df['b']
            return df
        run_and_compare(add, data=self.data)

    def test_mul_cst(self):

        def mul(df, **kwargs):
            return df * 2
        run_and_compare(mul, data=self.data)

    def test_mul_list(self):

        def mul(df, **kwargs):
            return df * [2, 3, 4, 5]
        run_and_compare(mul, data=self.data)

    @pytest.mark.parametrize('fill_value', fill_values)
    def test_mul_method_columns(self, fill_value):

        def mul1(df, fill_value, **kwargs):
            return df['a'].mul(df['b'], fill_value=fill_value)

        def mul2(df, fill_value, **kwargs):
            return df[['a', 'c']].mul(df[['b', 'a']], fill_value=fill_value)
        run_and_compare(mul1, data=self.data, fill_value=fill_value)
        run_and_compare(mul2, data=self.data, fill_value=fill_value)

    def test_mul_columns(self):

        def mul1(df, **kwargs):
            return df['a'] * df['b']

        def mul2(df, **kwargs):
            return df[['a', 'c']] * df[['b', 'a']]
        run_and_compare(mul1, data=self.data)
        run_and_compare(mul2, data=self.data)

    def test_mod_cst(self):

        def mod(df, **kwargs):
            return df % 2
        run_and_compare(mod, data=self.data)

    def test_mod_list(self):

        def mod(df, **kwargs):
            return df % [2, 3, 4, 5]
        run_and_compare(mod, data=self.data)

    @pytest.mark.parametrize('fill_value', fill_values)
    def test_mod_method_columns(self, fill_value):

        def mod1(df, fill_value, **kwargs):
            return df['a'].mod(df['b'], fill_value=fill_value)

        def mod2(df, fill_value, **kwargs):
            return df[['a', 'c']].mod(df[['b', 'a']], fill_value=fill_value)
        run_and_compare(mod1, data=self.data, fill_value=fill_value)
        run_and_compare(mod2, data=self.data, fill_value=fill_value)

    def test_mod_columns(self):

        def mod1(df, **kwargs):
            return df['a'] % df['b']

        def mod2(df, **kwargs):
            return df[['a', 'c']] % df[['b', 'a']]
        run_and_compare(mod1, data=self.data)
        run_and_compare(mod2, data=self.data)

    def test_truediv_cst(self):

        def truediv(df, **kwargs):
            return df / 2
        run_and_compare(truediv, data=self.data)

    def test_truediv_list(self):

        def truediv(df, **kwargs):
            return df / [1, 0.5, 0.2, 2.0]
        run_and_compare(truediv, data=self.data)

    @pytest.mark.parametrize('fill_value', fill_values)
    def test_truediv_method_columns(self, fill_value):

        def truediv1(df, fill_value, **kwargs):
            return df['a'].truediv(df['b'], fill_value=fill_value)

        def truediv2(df, fill_value, **kwargs):
            return df[['a', 'c']].truediv(df[['b', 'a']], fill_value=fill_value)
        run_and_compare(truediv1, data=self.data, fill_value=fill_value)
        run_and_compare(truediv2, data=self.data, fill_value=fill_value)

    def test_truediv_columns(self):

        def truediv1(df, **kwargs):
            return df['a'] / df['b']

        def truediv2(df, **kwargs):
            return df[['a', 'c']] / df[['b', 'a']]
        run_and_compare(truediv1, data=self.data)
        run_and_compare(truediv2, data=self.data)

    def test_floordiv_cst(self):

        def floordiv(df, **kwargs):
            return df // 2
        run_and_compare(floordiv, data=self.data)

    def test_floordiv_list(self):

        def floordiv(df, **kwargs):
            return df // [1, 0.54, 0.24, 2.01]
        run_and_compare(floordiv, data=self.data)

    @pytest.mark.parametrize('fill_value', fill_values)
    def test_floordiv_method_columns(self, fill_value):

        def floordiv1(df, fill_value, **kwargs):
            return df['a'].floordiv(df['b'], fill_value=fill_value)

        def floordiv2(df, fill_value, **kwargs):
            return df[['a', 'c']].floordiv(df[['b', 'a']], fill_value=fill_value)
        run_and_compare(floordiv1, data=self.data, fill_value=fill_value)
        run_and_compare(floordiv2, data=self.data, fill_value=fill_value)

    def test_floordiv_columns(self):

        def floordiv1(df, **kwargs):
            return df['a'] // df['b']

        def floordiv2(df, **kwargs):
            return df[['a', 'c']] // df[['b', 'a']]
        run_and_compare(floordiv1, data=self.data)
        run_and_compare(floordiv2, data=self.data)
    cmp_data = {'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50], 'c': [50.0, 40.0, 30.1, 20.0, 10.0]}
    cmp_fn_values = ['eq', 'ne', 'le', 'lt', 'ge', 'gt']

    @pytest.mark.parametrize('cmp_fn', cmp_fn_values)
    def test_cmp_cst(self, cmp_fn):

        def cmp1(df, cmp_fn, **kwargs):
            return getattr(df['a'], cmp_fn)(3)

        def cmp2(df, cmp_fn, **kwargs):
            return getattr(df, cmp_fn)(30)
        run_and_compare(cmp1, data=self.cmp_data, cmp_fn=cmp_fn)
        run_and_compare(cmp2, data=self.cmp_data, cmp_fn=cmp_fn)

    @pytest.mark.parametrize('cmp_fn', cmp_fn_values)
    def test_cmp_list(self, cmp_fn):

        def cmp(df, cmp_fn, **kwargs):
            return getattr(df, cmp_fn)([3, 30, 30.1])
        run_and_compare(cmp, data=self.cmp_data, cmp_fn=cmp_fn)

    @pytest.mark.parametrize('cmp_fn', cmp_fn_values)
    def test_cmp_cols(self, cmp_fn):

        def cmp1(df, cmp_fn, **kwargs):
            return getattr(df['b'], cmp_fn)(df['c'])

        def cmp2(df, cmp_fn, **kwargs):
            return getattr(df[['b', 'c']], cmp_fn)(df[['a', 'b']])
        run_and_compare(cmp1, data=self.cmp_data, cmp_fn=cmp_fn)
        run_and_compare(cmp2, data=self.cmp_data, cmp_fn=cmp_fn)

    @pytest.mark.parametrize('cmp_fn', cmp_fn_values)
    @pytest.mark.parametrize('value', [2, 2.2, 'a'])
    @pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
    def test_cmp_mixed_types(self, cmp_fn, value, data):

        def cmp(df, cmp_fn, value, **kwargs):
            return getattr(df, cmp_fn)(value)
        run_and_compare(cmp, data=data, cmp_fn=cmp_fn, value=value)

    def test_filter_dtypes(self):

        def filter(df, **kwargs):
            return df[df.a < 4].dtypes
        run_and_compare(filter, data=self.cmp_data)

    def test_filter_empty_result(self):

        def filter(df, **kwargs):
            return df[df.a < 0]
        run_and_compare(filter, data=self.cmp_data)

    def test_complex_filter(self):

        def filter_and(df, **kwargs):
            return df[(df.a < 5) & (df.b > 20)]

        def filter_or(df, **kwargs):
            return df[(df.a < 3) | (df.b > 40)]
        run_and_compare(filter_and, data=self.cmp_data)
        run_and_compare(filter_or, data=self.cmp_data)

    def test_string_bin_op(self):

        def test_bin_op(df, op_name, op_arg, **kwargs):
            return getattr(df, op_name)(op_arg)
        bin_ops = {'__add__': '_sfx', '__radd__': 'pref_', '__mul__': 10}
        for op, arg in bin_ops.items():
            run_and_compare(test_bin_op, data={'a': ['a']}, op_name=op, op_arg=arg, force_lazy=False)

    @pytest.mark.parametrize('force_hdk', [False, True])
    def test_arithmetic_ops(self, force_hdk):

        def compute(df, operation, **kwargs):
            df = getattr(df, operation)(3)
            return df
        for op in ('__add__', '__sub__', '__mul__', '__pow__', '__truediv__', '__floordiv__'):
            run_and_compare(compute, {'A': [1, 2, 3, 4, 5]}, operation=op, force_hdk_execute=force_hdk)

    @pytest.mark.parametrize('force_hdk', [False, True])
    def test_invert_op(self, force_hdk):

        def invert(df, **kwargs):
            return ~df
        run_and_compare(invert, {'A': [1, 2, 3, 4, 5]}, force_hdk_execute=force_hdk)