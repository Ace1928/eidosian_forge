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
def test_setitem_invalid_dtype(data):
    pa_type = data._pa_array.type
    if pa.types.is_string(pa_type) or pa.types.is_binary(pa_type):
        fill_value = 123
        err = TypeError
        msg = "Invalid value '123' for dtype"
    elif pa.types.is_integer(pa_type) or pa.types.is_floating(pa_type) or pa.types.is_boolean(pa_type):
        fill_value = 'foo'
        err = pa.ArrowInvalid
        msg = 'Could not convert'
    else:
        fill_value = 'foo'
        err = TypeError
        msg = "Invalid value 'foo' for dtype"
    with pytest.raises(err, match=msg):
        data[:] = fill_value