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
def test_roundtrip_timestamp(self, orient, convert_axes, datetime_frame):
    data = StringIO(datetime_frame.to_json(orient=orient))
    result = read_json(data, orient=orient, convert_axes=convert_axes)
    expected = datetime_frame.copy()
    if not convert_axes:
        idx = expected.index.view(np.int64) // 1000000
        if orient != 'split':
            idx = idx.astype(str)
        expected.index = idx
    assert_json_roundtrip_equal(result, expected, orient)