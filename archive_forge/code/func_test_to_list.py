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
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
def test_to_list(data):
    modin_series, pandas_series = create_test_series(data)
    pd_res = pandas_series.to_list()
    md_res = modin_series.to_list()
    assert type(pd_res) is type(md_res)
    assert np.array_equal(pd_res, md_res, equal_nan=True)