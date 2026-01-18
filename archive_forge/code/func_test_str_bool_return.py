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
def test_str_bool_return(self):
    index = Index(['a1', 'a2', 'b1', 'b2'])
    result = index.str.startswith('a')
    expected = np.array([True, True, False, False])
    tm.assert_numpy_array_equal(result, expected)
    assert isinstance(result, np.ndarray)