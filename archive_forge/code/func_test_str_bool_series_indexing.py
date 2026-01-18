from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
def test_str_bool_series_indexing(self):
    index = Index(['a1', 'a2', 'b1', 'b2'])
    s = Series(range(4), index=index)
    result = s[s.index.str.startswith('a')]
    expected = Series(range(2), index=['a1', 'a2'])
    tm.assert_series_equal(result, expected)