from typing import Any
import pandas as pd
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
from qpd_test.tests_base import TestsBase
from qpd_test.utils import assert_df_eq
from pytest import raises
def test_window_simple_agg_partition_by(self):

    def make_func(func, partition_keys, order_by, start=None, end=None):
        wtype = '' if start is None and end is None else 'rows'
        ws = WindowSpec('', partition_keys=partition_keys, order_by=OrderBySpec(*order_by), windowframe=make_windowframe_spec(wtype, start, end))
        return WindowFunctionSpec(func, False, False, ws)
    a = self.to_df([[0, 1], [None, 1], [0, None], [0, 2]], ['a', 'b'])
    self.w_eq(a, make_func('max', ['a'], ['b']), [col('b')], 'x', [[0, 1, 2], [None, 1, 1], [0, None, 2], [0, 2, 2]], ['a', 'b', 'x'])
    self.w_eq(a, make_func('max', ['a'], ['b'], start=-1, end=0), [col('b')], 'x', [[0, 1, 1], [None, 1, 1], [0, None, None], [0, 2, 2]], ['a', 'b', 'x'])
    self.w_eq(a, make_func('min', ['a'], ['b']), [col('b')], 'x', [[0, 1, 1], [None, 1, 1], [0, None, 1], [0, 2, 1]], ['a', 'b', 'x'])
    self.w_eq(a, make_func('min', ['a'], ['b'], start=-1, end=0), [col('b')], 'x', [[0, 1, 1], [None, 1, 1], [0, None, None], [0, 2, 1]], ['a', 'b', 'x'])
    self.w_eq(a, make_func('avg', ['a'], ['b']), [col('b')], 'x', [[0, 1, 1.5], [None, 1, 1], [0, None, 1.5], [0, 2, 1.5]], ['a', 'b', 'x'])
    self.w_eq(a, make_func('mean', ['a'], ['b'], start=-1, end=0), [col('b')], 'x', [[0, 1, 1.0], [None, 1, 1], [0, None, None], [0, 2, 1.5]], ['a', 'b', 'x'])
    self.w_eq(a, make_func('count', ['a'], ['b']), [col('b')], 'x', [[0, 1, 2], [None, 1, 1], [0, None, 2], [0, 2, 2]], ['a', 'b', 'x'])
    a = self.to_df([[0, 1], [None, 1], [0, None], [0, 3]], ['a', 'b'])
    self.w_eq(a, make_func('count', ['a'], [('b', True)], start=-1, end=0), [col('b')], 'x', [[0, 1, 1], [None, 1, 1], [0, None, 0], [0, 3, 2]], ['a', 'b', 'x'])