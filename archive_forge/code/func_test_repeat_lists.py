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
@pytest.mark.parametrize('data', [np.arange(256)])
@pytest.mark.parametrize('repeats', [0, 2, [2], np.arange(256), [0] * 64 + [2] * 64 + [3] * 32 + [4] * 32 + [5] * 64, [2] * 257], ids=['0_case', 'scalar', 'one-elem-list', 'array', 'list', 'wrong_list'])
def test_repeat_lists(data, repeats, request):
    expected_exception = None
    if 'wrong_list' in request.node.callspec.id:
        expected_exception = ValueError('operands could not be broadcast together with shape (256,) (257,)')
    eval_general(*create_test_series(data), lambda df: df.repeat(repeats), expected_exception=expected_exception)