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
def test_union_dt_as_obj(self, simple_index):
    index = simple_index
    date_index = date_range('2019-01-01', periods=10)
    first_cat = index.union(date_index)
    second_cat = index.union(index)
    appended = Index(np.append(index, date_index.astype('O')))
    tm.assert_index_equal(first_cat, appended)
    tm.assert_index_equal(second_cat, index)
    tm.assert_contains_all(index, first_cat)
    tm.assert_contains_all(index, second_cat)
    tm.assert_contains_all(date_index, first_cat)