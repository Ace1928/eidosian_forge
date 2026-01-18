from __future__ import annotations
import datetime
import itertools
import json
import unittest.mock as mock
import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
from pandas.core.indexing import IndexingError
from pandas.errors import SpecificationError
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.pandas.testing import assert_series_equal
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution, try_cast_to_pandas
from .utils import (
def test_values_ea():
    data = pandas.arrays.SparseArray(np.arange(10, dtype='int64'))
    modin_series, pandas_series = create_test_series(data)
    modin_values = modin_series.values
    pandas_values = pandas_series.values
    assert modin_values.dtype == pandas_values.dtype
    df_equals(modin_values, pandas_values)