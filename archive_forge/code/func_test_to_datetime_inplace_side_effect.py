import contextlib
import numpy as np
import pandas
import pytest
from numpy.testing import assert_array_equal
import modin.pandas as pd
from modin.config import StorageFormat
from modin.pandas.io import to_pandas
from modin.pandas.testing import assert_frame_equal
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
from .utils import (
def test_to_datetime_inplace_side_effect():
    times = list(range(1617993360, 1618193360))
    values = list(range(215441, 415441))
    modin_df = pd.DataFrame({'time': times, 'value': values})
    pandas_df = pandas.DataFrame({'time': times, 'value': values})
    df_equals(pd.to_datetime(modin_df['time'], unit='s'), pandas.to_datetime(pandas_df['time'], unit='s'))