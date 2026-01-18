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
def test_to_json_multiindex_escape(self):
    df = DataFrame(True, index=date_range('2017-01-20', '2017-01-23'), columns=['foo', 'bar']).stack(future_stack=True)
    result = df.to_json()
    expected = '{"(Timestamp(\'2017-01-20 00:00:00\'), \'foo\')":true,"(Timestamp(\'2017-01-20 00:00:00\'), \'bar\')":true,"(Timestamp(\'2017-01-21 00:00:00\'), \'foo\')":true,"(Timestamp(\'2017-01-21 00:00:00\'), \'bar\')":true,"(Timestamp(\'2017-01-22 00:00:00\'), \'foo\')":true,"(Timestamp(\'2017-01-22 00:00:00\'), \'bar\')":true,"(Timestamp(\'2017-01-23 00:00:00\'), \'foo\')":true,"(Timestamp(\'2017-01-23 00:00:00\'), \'bar\')":true}'
    assert result == expected