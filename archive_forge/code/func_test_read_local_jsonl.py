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
def test_read_local_jsonl(self):
    with tm.ensure_clean('tmp_items.json') as path:
        with open(path, 'w', encoding='utf-8') as infile:
            infile.write('{"a": 1, "b": 2}\n{"b":2, "a" :1}\n')
        result = read_json(path, lines=True)
        expected = DataFrame([[1, 2], [1, 2]], columns=['a', 'b'])
        tm.assert_frame_equal(result, expected)