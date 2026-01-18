import string
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('kind', ['line', 'area'])
def test_subplots_timeseries_rot(self, kind):
    idx = date_range(start='2014-07-01', freq='ME', periods=10)
    df = DataFrame(np.random.default_rng(2).random((10, 3)), index=idx)
    axes = df.plot(kind=kind, subplots=True, sharex=False, rot=45, fontsize=7)
    for ax in axes:
        _check_visible(ax.xaxis)
        _check_visible(ax.get_xticklabels())
        _check_visible(ax.get_xticklabels(minor=True))
        _check_visible(ax.xaxis.get_label())
        _check_visible(ax.get_yticklabels())
        _check_ticks_props(ax, xlabelsize=7, xrot=45, ylabelsize=7)