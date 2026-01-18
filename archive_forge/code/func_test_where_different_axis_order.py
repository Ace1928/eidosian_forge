import warnings
import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions, RangePartitioning, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
def test_where_different_axis_order():
    data = test_data['float_nan_data']
    pandas_df = pandas.DataFrame(data)
    pandas_cond_df = pandas_df % 5 < 2
    pandas_cond_df = pandas_cond_df.reindex(columns=pandas_df.columns[::-1], index=pandas_df.index[::-1])
    pandas_other_df = -pandas_df
    pandas_other_df = pandas_other_df.reindex(columns=pandas_df.columns[-1:].append(pandas_df.columns[:-1]), index=pandas_df.index[-1:].append(pandas_df.index[:-1]))
    modin_df = pd.DataFrame(pandas_df)
    modin_cond_df = pd.DataFrame(pandas_cond_df)
    modin_other_df = pd.DataFrame(pandas_other_df)
    pandas_result = pandas_df.where(pandas_cond_df, pandas_other_df)
    modin_result = modin_df.where(modin_cond_df, modin_other_df)
    df_equals(modin_result, pandas_result)