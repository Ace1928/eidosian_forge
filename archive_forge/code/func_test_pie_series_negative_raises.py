from datetime import datetime
from itertools import chain
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_pie_series_negative_raises(self):
    series = Series([1, 2, 0, 4, -1], index=['a', 'b', 'c', 'd', 'e'])
    with pytest.raises(ValueError, match="pie plot doesn't allow negative values"):
        series.plot.pie()