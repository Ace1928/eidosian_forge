from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
def test_sum_corner(self):
    empty_frame = DataFrame()
    axis0 = empty_frame.sum(0)
    axis1 = empty_frame.sum(1)
    assert isinstance(axis0, Series)
    assert isinstance(axis1, Series)
    assert len(axis0) == 0
    assert len(axis1) == 0