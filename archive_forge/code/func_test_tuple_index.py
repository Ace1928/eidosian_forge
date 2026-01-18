from datetime import timedelta
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
from pandas import (
import pandas._testing as tm
def test_tuple_index():
    s = Series([1, 2], index=[('a',), ('b',)])
    assert s['a',] == 1
    assert s['b',] == 2
    s['b',] = 3
    assert s['b',] == 3