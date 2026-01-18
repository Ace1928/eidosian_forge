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
def test_rename_invalid(self):
    df = self.df([['a', 1]], 'a:str,b:int')
    raises(FugueDataFrameOperationError, lambda: fi.rename(df, columns=dict(aa='ab')))