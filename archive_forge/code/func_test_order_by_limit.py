from typing import Any
import pandas as pd
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
from qpd_test.tests_base import TestsBase
from qpd_test.utils import assert_df_eq
from pytest import raises
def test_order_by_limit(self):

    def assert_eq(df, order_by, limit, expected, expected_cols):
        res = self.to_pandas_df(self.op.order_by_limit(df, OrderBySpec(*order_by), limit))
        assert_df_eq(res, expected, expected_cols)
    a = self.to_df([['x', 'a'], ['x', 'a'], [None, None]], ['a', 'b'])
    assert_eq(a, ['a'], 1, [[None, None]], ['a', 'b'])
    assert_eq(a, ['a'], 2, [[None, None], ['x', 'a']], ['a', 'b'])
    assert_eq(a, [('a', True, 'last')], 2, [['x', 'a'], ['x', 'a']], ['a', 'b'])
    assert_eq(a, [('a', False, 'first')], 2, [['x', 'a'], [None, None]], ['a', 'b'])