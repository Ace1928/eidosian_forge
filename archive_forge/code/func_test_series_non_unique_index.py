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
def test_series_non_unique_index(self):
    s = Series(['a', 'b'], index=[1, 1])
    msg = "Series index must be unique for orient='index'"
    with pytest.raises(ValueError, match=msg):
        s.to_json(orient='index')
    tm.assert_series_equal(s, read_json(StringIO(s.to_json(orient='split')), orient='split', typ='series'))
    unserialized = read_json(StringIO(s.to_json(orient='records')), orient='records', typ='series')
    tm.assert_equal(s.values, unserialized.values)