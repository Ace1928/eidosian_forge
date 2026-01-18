import copy
import os
import pickle
from datetime import datetime
from unittest import TestCase
import pandas as pd
import pytest
from pytest import raises
from triad.exceptions import InvalidOperationError
from triad.utils.io import isfile, makedirs, touch
import fugue.api as fa
import fugue.column.functions as ff
from fugue import (
from fugue.column import all_cols, col, lit
from fugue.dataframe.utils import _df_eq as df_eq
from fugue.execution.native_execution_engine import NativeExecutionEngine
def test_to_df_general(self):
    e = self.engine
    o = ArrayDataFrame([[1.1, 2.2], [3.3, 4.4]], 'a:double,b:double')
    df_eq(o, fa.as_fugue_engine_df(e, o), throw=True)
    df_eq(o, fa.as_fugue_engine_df(e, [[1.1, 2.2], [3.3, 4.4]], 'a:double,b:double'), throw=True)
    pdf = pd.DataFrame([[1.1, 2.2], [3.3, 4.4]], columns=['a', 'b'])
    df_eq(o, fa.as_fugue_engine_df(e, pdf), throw=True)
    df_eq(fa.as_fugue_engine_df(e, [['2020-01-01']], 'a:datetime'), [[datetime(2020, 1, 1)]], 'a:datetime', throw=True)
    o = ArrayDataFrame([], 'a:double,b:str')
    pdf = pd.DataFrame([[0.1, 'a']], columns=['a', 'b'])
    pdf = pdf[pdf.a < 0]
    df_eq(o, fa.as_fugue_engine_df(e, pdf), throw=True)