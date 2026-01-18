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
@pytest.mark.parametrize('kwargs', [{}, {'secondary_y': True}])
def test_ts_line_lim(self, ts, kwargs):
    _, ax = mpl.pyplot.subplots()
    ax = ts.plot(ax=ax, **kwargs)
    xmin, xmax = ax.get_xlim()
    lines = ax.get_lines()
    assert xmin <= lines[0].get_data(orig=False)[0][0]
    assert xmax >= lines[0].get_data(orig=False)[0][-1]