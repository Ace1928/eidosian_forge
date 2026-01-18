from collections import (
from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
def test_to_dict_masked_native_python(self):
    df = DataFrame({'a': Series([1, 2], dtype='Int64'), 'B': 1})
    result = df.to_dict(orient='records')
    assert isinstance(result[0]['a'], int)
    df = DataFrame({'a': Series([1, NA], dtype='Int64'), 'B': 1})
    result = df.to_dict(orient='records')
    assert isinstance(result[0]['a'], int)