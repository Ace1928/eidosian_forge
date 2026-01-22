from __future__ import annotations
from functools import reduce
from itertools import product
import operator
import numpy as np
import pytest
from pandas.compat import PY312
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation import (
from pandas.core.computation.engines import ENGINES
from pandas.core.computation.expr import (
from pandas.core.computation.expressions import (
from pandas.core.computation.ops import (
from pandas.core.computation.scope import DEFAULT_GLOBALS
class TestMath:

    def eval(self, *args, **kwargs):
        kwargs['level'] = kwargs.pop('level', 0) + 1
        return pd.eval(*args, **kwargs)

    @pytest.mark.skipif(not NUMEXPR_INSTALLED, reason='Unary ops only implemented for numexpr')
    @pytest.mark.parametrize('fn', _unary_math_ops)
    def test_unary_functions(self, fn):
        df = DataFrame({'a': np.random.default_rng(2).standard_normal(10)})
        a = df.a
        expr = f'{fn}(a)'
        got = self.eval(expr)
        with np.errstate(all='ignore'):
            expect = getattr(np, fn)(a)
        tm.assert_series_equal(got, expect, check_names=False)

    @pytest.mark.parametrize('fn', _binary_math_ops)
    def test_binary_functions(self, fn):
        df = DataFrame({'a': np.random.default_rng(2).standard_normal(10), 'b': np.random.default_rng(2).standard_normal(10)})
        a = df.a
        b = df.b
        expr = f'{fn}(a, b)'
        got = self.eval(expr)
        with np.errstate(all='ignore'):
            expect = getattr(np, fn)(a, b)
        tm.assert_almost_equal(got, expect, check_names=False)

    def test_df_use_case(self, engine, parser):
        df = DataFrame({'a': np.random.default_rng(2).standard_normal(10), 'b': np.random.default_rng(2).standard_normal(10)})
        df.eval('e = arctan2(sin(a), b)', engine=engine, parser=parser, inplace=True)
        got = df.e
        expect = np.arctan2(np.sin(df.a), df.b)
        tm.assert_series_equal(got, expect, check_names=False)

    def test_df_arithmetic_subexpression(self, engine, parser):
        df = DataFrame({'a': np.random.default_rng(2).standard_normal(10), 'b': np.random.default_rng(2).standard_normal(10)})
        df.eval('e = sin(a + b)', engine=engine, parser=parser, inplace=True)
        got = df.e
        expect = np.sin(df.a + df.b)
        tm.assert_series_equal(got, expect, check_names=False)

    @pytest.mark.parametrize('dtype, expect_dtype', [(np.int32, np.float64), (np.int64, np.float64), (np.float32, np.float32), (np.float64, np.float64), pytest.param(np.complex128, np.complex128, marks=td.skip_if_windows)])
    def test_result_types(self, dtype, expect_dtype, engine, parser):
        df = DataFrame({'a': np.random.default_rng(2).standard_normal(10).astype(dtype)})
        assert df.a.dtype == dtype
        df.eval('b = sin(a)', engine=engine, parser=parser, inplace=True)
        got = df.b
        expect = np.sin(df.a)
        assert expect.dtype == got.dtype
        assert expect_dtype == got.dtype
        tm.assert_series_equal(got, expect, check_names=False)

    def test_undefined_func(self, engine, parser):
        df = DataFrame({'a': np.random.default_rng(2).standard_normal(10)})
        msg = '"mysin" is not a supported function'
        with pytest.raises(ValueError, match=msg):
            df.eval('mysin(a)', engine=engine, parser=parser)

    def test_keyword_arg(self, engine, parser):
        df = DataFrame({'a': np.random.default_rng(2).standard_normal(10)})
        msg = 'Function "sin" does not support keyword arguments'
        with pytest.raises(TypeError, match=msg):
            df.eval('sin(x=a)', engine=engine, parser=parser)