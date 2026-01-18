from datetime import (
import inspect
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
def test_set_reset_index_intervalindex(self):
    df = DataFrame({'A': range(10)})
    ser = pd.cut(df.A, 5)
    df['B'] = ser
    df = df.set_index('B')
    df = df.reset_index()