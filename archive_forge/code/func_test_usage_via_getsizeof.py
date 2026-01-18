from io import StringIO
import re
from string import ascii_uppercase as uppercase
import sys
import textwrap
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
@pytest.mark.skipif(PYPY, reason='PyPy getsizeof() fails by design')
def test_usage_via_getsizeof():
    df = DataFrame(data=1, index=MultiIndex.from_product([['a'], range(1000)]), columns=['A'])
    mem = df.memory_usage(deep=True).sum()
    diff = mem - sys.getsizeof(df)
    assert abs(diff) < 100