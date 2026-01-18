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
def test_to_json_series_of_objects(self):

    class _TestObject:

        def __init__(self, a, b, _c, d) -> None:
            self.a = a
            self.b = b
            self._c = _c
            self.d = d

        def e(self):
            return 5
    series = Series([_TestObject(a=1, b=2, _c=3, d=4)])
    assert json.loads(series.to_json()) == {'0': {'a': 1, 'b': 2, 'd': 4}}