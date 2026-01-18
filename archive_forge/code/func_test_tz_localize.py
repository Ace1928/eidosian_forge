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
def test_tz_localize():
    idx = pd.date_range('1/1/2012', periods=400, freq='2D')
    data = np.random.randint(0, 100, size=(len(idx), 4))
    modin_df = pd.DataFrame(data, index=idx)
    pandas_df = pandas.DataFrame(data, index=idx)
    df_equals(modin_df.tz_localize('UTC', axis=0), pandas_df.tz_localize('UTC', axis=0))
    df_equals(modin_df.tz_localize('America/Los_Angeles', axis=0), pandas_df.tz_localize('America/Los_Angeles', axis=0))