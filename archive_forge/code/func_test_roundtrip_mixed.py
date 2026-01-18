import datetime
from datetime import timedelta
from decimal import Decimal
from io import (
import json
import os
import sys
import time
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import IS64
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.io.json import ujson_dumps
@pytest.mark.parametrize('convert_axes', [True, False])
def test_roundtrip_mixed(self, orient, convert_axes):
    index = Index(['a', 'b', 'c', 'd', 'e'])
    values = {'A': [0.0, 1.0, 2.0, 3.0, 4.0], 'B': [0.0, 1.0, 0.0, 1.0, 0.0], 'C': ['foo1', 'foo2', 'foo3', 'foo4', 'foo5'], 'D': [True, False, True, False, True]}
    df = DataFrame(data=values, index=index)
    data = StringIO(df.to_json(orient=orient))
    result = read_json(data, orient=orient, convert_axes=convert_axes)
    expected = df.copy()
    expected = expected.assign(**expected.select_dtypes('number').astype(np.int64))
    assert_json_roundtrip_equal(result, expected, orient)