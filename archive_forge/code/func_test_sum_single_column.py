import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.testing import assert_series_equal
from modin.tests.pandas.utils import (
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
def test_sum_single_column(data):
    modin_df = pd.DataFrame(data).iloc[:, [0]]
    pandas_df = pandas.DataFrame(data).iloc[:, [0]]
    df_equals(modin_df.sum(), pandas_df.sum())
    df_equals(modin_df.sum(axis=1), pandas_df.sum(axis=1))