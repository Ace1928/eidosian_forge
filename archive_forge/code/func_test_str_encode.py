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
@pytest.mark.parametrize('encoding', encoding_types)
@pytest.mark.parametrize('errors', ['strict', 'ignore', 'replace'])
def test_str_encode(encoding, errors, str_encode_decode_test_data):
    expected_exception = None
    if errors == 'strict' and encoding == 'ascii':
        expected_exception = False
    eval_general(*create_test_series(str_encode_decode_test_data), lambda s: s.str.encode(encoding, errors=errors), expected_exception=expected_exception)