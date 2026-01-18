from datetime import (
import gc
import itertools
import re
import string
import weakref
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('x, y', [('dates', 'vals'), (0, 1)])
def test_scatterplot_datetime_data(self, x, y):
    dates = date_range(start=date(2019, 1, 1), periods=12, freq='W')
    vals = np.random.default_rng(2).normal(0, 1, len(dates))
    df = DataFrame({'dates': dates, 'vals': vals})
    _check_plot_works(df.plot.scatter, x=x, y=y)