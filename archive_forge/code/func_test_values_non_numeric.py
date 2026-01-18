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
def test_values_non_numeric():
    data = ['str{0}'.format(i) for i in range(0, 10 ** 3)]
    modin_series, pandas_series = create_test_series(data)
    modin_series = modin_series.astype('category')
    pandas_series = pandas_series.astype('category')
    df_equals(modin_series.values, pandas_series.values)