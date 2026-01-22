from datetime import (
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED
class TestSeriesFlexArithmetic:

    @pytest.mark.parametrize('ts', [(lambda x: x, lambda x: x * 2, False), (lambda x: x, lambda x: x[::2], False), (lambda x: x, lambda x: 5, True), (lambda x: Series(range(10), dtype=np.float64), lambda x: Series(range(10), dtype=np.float64), True)])
    @pytest.mark.parametrize('opname', ['add', 'sub', 'mul', 'floordiv', 'truediv', 'pow'])
    def test_flex_method_equivalence(self, opname, ts):
        tser = Series(np.arange(20, dtype=np.float64), index=date_range('2020-01-01', periods=20), name='ts')
        series = ts[0](tser)
        other = ts[1](tser)
        check_reverse = ts[2]
        op = getattr(Series, opname)
        alt = getattr(operator, opname)
        result = op(series, other)
        expected = alt(series, other)
        tm.assert_almost_equal(result, expected)
        if check_reverse:
            rop = getattr(Series, 'r' + opname)
            result = rop(series, other)
            expected = alt(other, series)
            tm.assert_almost_equal(result, expected)

    def test_flex_method_subclass_metadata_preservation(self, all_arithmetic_operators):

        class MySeries(Series):
            _metadata = ['x']

            @property
            def _constructor(self):
                return MySeries
        opname = all_arithmetic_operators
        op = getattr(Series, opname)
        m = MySeries([1, 2, 3], name='test')
        m.x = 42
        result = op(m, 1)
        assert result.x == 42

    def test_flex_add_scalar_fill_value(self):
        ser = Series([0, 1, np.nan, 3, 4, 5])
        exp = ser.fillna(0).add(2)
        res = ser.add(2, fill_value=0)
        tm.assert_series_equal(res, exp)
    pairings = [(Series.div, operator.truediv, 1), (Series.rdiv, ops.rtruediv, 1)]
    for op in ['add', 'sub', 'mul', 'pow', 'truediv', 'floordiv']:
        fv = 0
        lop = getattr(Series, op)
        lequiv = getattr(operator, op)
        rop = getattr(Series, 'r' + op)
        requiv = lambda x, y, op=op: getattr(operator, op)(y, x)
        pairings.append((lop, lequiv, fv))
        pairings.append((rop, requiv, fv))

    @pytest.mark.parametrize('op, equiv_op, fv', pairings)
    def test_operators_combine(self, op, equiv_op, fv):

        def _check_fill(meth, op, a, b, fill_value=0):
            exp_index = a.index.union(b.index)
            a = a.reindex(exp_index)
            b = b.reindex(exp_index)
            amask = isna(a)
            bmask = isna(b)
            exp_values = []
            for i in range(len(exp_index)):
                with np.errstate(all='ignore'):
                    if amask[i]:
                        if bmask[i]:
                            exp_values.append(np.nan)
                            continue
                        exp_values.append(op(fill_value, b[i]))
                    elif bmask[i]:
                        if amask[i]:
                            exp_values.append(np.nan)
                            continue
                        exp_values.append(op(a[i], fill_value))
                    else:
                        exp_values.append(op(a[i], b[i]))
            result = meth(a, b, fill_value=fill_value)
            expected = Series(exp_values, exp_index)
            tm.assert_series_equal(result, expected)
        a = Series([np.nan, 1.0, 2.0, 3.0, np.nan], index=np.arange(5))
        b = Series([np.nan, 1, np.nan, 3, np.nan, 4.0], index=np.arange(6))
        result = op(a, b)
        exp = equiv_op(a, b)
        tm.assert_series_equal(result, exp)
        _check_fill(op, equiv_op, a, b, fill_value=fv)
        op(a, b, axis=0)