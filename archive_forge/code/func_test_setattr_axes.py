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
def test_setattr_axes():
    df = pd.DataFrame([[1, 2], [3, 4]])
    with warnings.catch_warnings():
        if get_current_execution() != 'BaseOnPython':
            warnings.simplefilter('error')
        if StorageFormat.get() != 'Hdk':
            df.index = ['foo', 'bar']
            pd.testing.assert_index_equal(df.index, pandas.Index(['foo', 'bar']))
        df.columns = [9, 10]
        pd.testing.assert_index_equal(df.columns, pandas.Index([9, 10]))