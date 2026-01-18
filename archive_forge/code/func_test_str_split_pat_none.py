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
@pytest.mark.parametrize('method', ['rsplit', 'split'])
def test_str_split_pat_none(method):
    ser = pd.Series(['a1 cbc\nb', None], dtype=ArrowDtype(pa.string()))
    result = getattr(ser.str, method)()
    expected = pd.Series(ArrowExtensionArray(pa.array([['a1', 'cbc', 'b'], None])))
    tm.assert_series_equal(result, expected)