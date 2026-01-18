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
@pytest.mark.parametrize('side, pat, na, exp', [['startswith', 'ab', None, [True, None, False]], ['startswith', 'b', False, [False, False, False]], ['endswith', 'b', True, [False, True, False]], ['endswith', 'bc', None, [True, None, False]], ['startswith', ('a', 'e', 'g'), None, [True, None, True]], ['endswith', ('a', 'c', 'g'), None, [True, None, True]], ['startswith', (), None, [False, None, False]], ['endswith', (), None, [False, None, False]]])
def test_str_start_ends_with(side, pat, na, exp):
    ser = pd.Series(['abc', None, 'efg'], dtype=ArrowDtype(pa.string()))
    result = getattr(ser.str, side)(pat, na=na)
    expected = pd.Series(exp, dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)