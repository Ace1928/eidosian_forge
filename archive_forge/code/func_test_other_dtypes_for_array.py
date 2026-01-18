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
@pytest.mark.parametrize('func', ['is_datetime_array', 'is_datetime64_array', 'is_bool_array', 'is_timedelta_or_timedelta64_array', 'is_date_array', 'is_time_array', 'is_interval_array'])
def test_other_dtypes_for_array(self, func):
    func = getattr(lib, func)
    arr = np.array(['foo', 'bar'])
    assert not func(arr)
    assert not func(arr.reshape(2, 1))
    arr = np.array([1, 2])
    assert not func(arr)
    assert not func(arr.reshape(2, 1))