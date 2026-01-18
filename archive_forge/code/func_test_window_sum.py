from typing import Any
import pandas as pd
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
from qpd_test.tests_base import TestsBase
from qpd_test.utils import assert_df_eq
from pytest import raises
def test_window_sum(self):

    def make_func(partition_keys, order_by, start=None, end=None):
        wtype = '' if start is None and end is None else 'rows'
        ws = WindowSpec('', partition_keys=partition_keys, order_by=OrderBySpec(*order_by), windowframe=make_windowframe_spec(wtype, start, end))
        return WindowFunctionSpec('sum', False, False, ws)
    a = self.to_df([[0, 1], [0, 2]], ['a', 'b'])
    self.w_eq(a, make_func([], []), [col('b')], 'x', [[0, 1, 3], [0, 2, 3]], ['a', 'b', 'x'])
    self.w_eq(a, make_func([], [('a', True), ('b', False)]), [col('b')], 'x', [[0, 1, 3], [0, 2, 3]], ['a', 'b', 'x'])
    a = self.to_df([[0, 1], [None, 1], [0, None], [0, 2]], ['a', 'b'])
    self.w_eq(a, make_func([], ['b', 'a'], end=0), [col('b')], 'x', [[0, 1, 2], [None, 1, 1], [0, None, None], [0, 2, 4]], ['a', 'b', 'x'])
    self.w_eq(a, make_func([], ['b', 'a'], start=-1, end=0), [col('b')], 'x', [[0, 1, 2], [None, 1, 1], [0, None, None], [0, 2, 3]], ['a', 'b', 'x'])
    if not pd.__version__.startswith('1.1'):
        self.w_eq(a, make_func([], ['b', 'a'], end=-1), [col('b')], 'x', [[0, 1, 1], [None, 1, None], [0, None, None], [0, 2, 2]], ['a', 'b', 'x'])
        self.w_eq(a, make_func([], ['b', 'a'], start=-2, end=-1), [col('b')], 'x', [[0, 1, 1], [None, 1, None], [0, None, None], [0, 2, 2]], ['a', 'b', 'x'])
        self.w_eq(a, make_func([], ['b', 'a'], start=1, end=None), [col('b')], 'x', [[0, 1, 2], [None, 1, 3], [0, None, 4], [0, 2, None]], ['a', 'b', 'x'])