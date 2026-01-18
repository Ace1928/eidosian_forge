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
def test_ts_tz_area_lim_xcompat_secondary_y(self, ts):
    tz_ts = ts.copy()
    tz_ts.index = tz_ts.tz_localize('GMT').tz_convert('CET')
    _, ax = mpl.pyplot.subplots()
    ax = tz_ts.plot.area(stacked=False, secondary_y=True, ax=ax)
    xmin, xmax = ax.get_xlim()
    line = ax.get_lines()[0].get_data(orig=False)[0]
    assert xmin <= line[0]
    assert xmax >= line[-1]
    _check_ticks_props(ax, xrot=0)