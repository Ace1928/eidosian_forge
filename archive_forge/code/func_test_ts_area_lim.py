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
def test_ts_area_lim(self, ts):
    _, ax = mpl.pyplot.subplots()
    ax = ts.plot.area(stacked=False, ax=ax)
    xmin, xmax = ax.get_xlim()
    line = ax.get_lines()[0].get_data(orig=False)[0]
    assert xmin <= line[0]
    assert xmax >= line[-1]
    _check_ticks_props(ax, xrot=0)