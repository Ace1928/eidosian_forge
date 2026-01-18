from __future__ import annotations
from datetime import (
from decimal import Decimal
from io import (
import operator
import pickle
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import timezones
from pandas.compat import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import (
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import no_default
from pandas.api.types import (
from pandas.tests.extension import base
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.arrow.extension_types import ArrowPeriodType
@pytest.mark.parametrize('dtype', ['string', 'string[pyarrow]'])
def test_series_from_string_array(dtype):
    arr = pa.array('the quick brown fox'.split())
    ser = pd.Series(arr, dtype=dtype)
    expected = pd.Series(ArrowExtensionArray(arr), dtype=dtype)
    tm.assert_series_equal(ser, expected)