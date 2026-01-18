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
@pytest.mark.parametrize('expand,expected', [(None, Index([['a', 'b', 'c'], ['d', 'e'], ['f']])), (False, Index([['a', 'b', 'c'], ['d', 'e'], ['f']])), (True, MultiIndex.from_tuples([('a', 'b', 'c'), ('d', 'e', np.nan), ('f', np.nan, np.nan)]))])
def test_str_split(self, expand, expected):
    index = Index(['a b c', 'd e', 'f'])
    if expand is not None:
        result = index.str.split(expand=expand)
    else:
        result = index.str.split()
    tm.assert_index_equal(result, expected)