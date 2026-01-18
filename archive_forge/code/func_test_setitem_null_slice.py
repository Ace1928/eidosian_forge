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
def test_setitem_null_slice(data):
    orig = data.copy()
    result = orig.copy()
    result[:] = data[0]
    expected = ArrowExtensionArray._from_sequence([data[0]] * len(data), dtype=data.dtype)
    tm.assert_extension_array_equal(result, expected)
    result = orig.copy()
    result[:] = data[::-1]
    expected = data[::-1]
    tm.assert_extension_array_equal(result, expected)
    result = orig.copy()
    result[:] = data.tolist()
    expected = data
    tm.assert_extension_array_equal(result, expected)