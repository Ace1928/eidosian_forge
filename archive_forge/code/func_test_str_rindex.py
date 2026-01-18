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
@pytest.mark.parametrize('data', test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize('sub', string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize('start, end', [(0, None), (1, -1), (1, 3)], ids=['default', 'non_default_working', 'exception'])
def test_str_rindex(data, sub, start, end, request):
    modin_series, pandas_series = create_test_series(data)
    expected_exception = None
    if 'exception-comma sep' in request.node.callspec.id:
        expected_exception = ValueError('substring not found')
    eval_general(modin_series, pandas_series, lambda series: series.str.rindex(sub, start=start, end=end), expected_exception=expected_exception)