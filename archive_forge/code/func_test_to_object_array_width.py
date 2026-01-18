import collections
from collections import namedtuple
from collections.abc import Iterator
from datetime import (
from decimal import Decimal
from fractions import Fraction
from io import StringIO
import itertools
from numbers import Number
import re
import sys
from typing import (
import numpy as np
import pytest
import pytz
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.core.dtypes import inference
from pandas.core.dtypes.cast import find_result_type
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_to_object_array_width(self):
    rows = [[1, 2, 3], [4, 5, 6]]
    expected = np.array(rows, dtype=object)
    out = lib.to_object_array(rows)
    tm.assert_numpy_array_equal(out, expected)
    expected = np.array(rows, dtype=object)
    out = lib.to_object_array(rows, min_width=1)
    tm.assert_numpy_array_equal(out, expected)
    expected = np.array([[1, 2, 3, None, None], [4, 5, 6, None, None]], dtype=object)
    out = lib.to_object_array(rows, min_width=5)
    tm.assert_numpy_array_equal(out, expected)