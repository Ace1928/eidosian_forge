from datetime import date, datetime
from typing import Any
from unittest import TestCase
import numpy as np
import pandas as pd
from pytest import raises
import fugue.api as fi
from fugue.dataframe import ArrowDataFrame, DataFrame
from fugue.dataframe.utils import _df_eq as df_eq
from fugue.exceptions import FugueDataFrameOperationError, FugueDatasetEmptyError
class NativeTests(Tests):

    def to_native_df(self, pdf: pd.DataFrame) -> Any:
        raise NotImplementedError

    def test_get_column_names(self):
        df = self.to_native_df(pd.DataFrame([[0, 1, 2]], columns=['0', '1', '2']))
        assert fi.get_column_names(df) == ['0', '1', '2']

    def test_rename_any_names(self):
        pdf = self.to_native_df(pd.DataFrame([[0, 1, 2]], columns=['a', 'b', 'c']))
        df = fi.rename(pdf, {})
        assert fi.get_column_names(df) == ['a', 'b', 'c']
        pdf = self.to_native_df(pd.DataFrame([[0, 1, 2]], columns=['0', '1', '2']))
        df = fi.rename(pdf, {'0': '_0', '1': '_1', '2': '_2'})
        assert fi.get_column_names(df) == ['_0', '_1', '_2']