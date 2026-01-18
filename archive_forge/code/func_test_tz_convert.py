import io
import warnings
import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
def test_tz_convert():
    modin_idx = pd.date_range('1/1/2012', periods=500, freq='2D', tz='America/Los_Angeles')
    pandas_idx = pandas.date_range('1/1/2012', periods=500, freq='2D', tz='America/Los_Angeles')
    data = np.random.randint(0, 100, size=(len(modin_idx), 4))
    modin_df = pd.DataFrame(data, index=modin_idx)
    pandas_df = pandas.DataFrame(data, index=pandas_idx)
    modin_result = modin_df.tz_convert('UTC', axis=0)
    pandas_result = pandas_df.tz_convert('UTC', axis=0)
    df_equals(modin_result, pandas_result)
    modin_multi = pd.MultiIndex.from_arrays([modin_idx, range(len(modin_idx))])
    pandas_multi = pandas.MultiIndex.from_arrays([pandas_idx, range(len(modin_idx))])
    modin_series = pd.DataFrame(data, index=modin_multi)
    pandas_series = pandas.DataFrame(data, index=pandas_multi)
    df_equals(modin_series.tz_convert('UTC', axis=0, level=0), pandas_series.tz_convert('UTC', axis=0, level=0))