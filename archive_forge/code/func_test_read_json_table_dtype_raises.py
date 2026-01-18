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
@pytest.mark.parametrize('dtype', [True, {'b': int, 'c': int}])
def test_read_json_table_dtype_raises(self, dtype):
    df = DataFrame({'a': [1, 2], 'b': [3.0, 4.0], 'c': ['5', '6']})
    dfjson = df.to_json(orient='table')
    msg = "cannot pass both dtype and orient='table'"
    with pytest.raises(ValueError, match=msg):
        read_json(dfjson, orient='table', dtype=dtype)