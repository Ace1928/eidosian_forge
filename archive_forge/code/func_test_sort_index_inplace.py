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
@pytest.mark.parametrize('axis', ['rows', 'columns'])
def test_sort_index_inplace(axis):
    data = test_data['int_data']
    modin_df, pandas_df = (pd.DataFrame(data), pandas.DataFrame(data))
    for df in [modin_df, pandas_df]:
        df.sort_index(axis=axis, inplace=True)
    df_equals(modin_df, pandas_df)