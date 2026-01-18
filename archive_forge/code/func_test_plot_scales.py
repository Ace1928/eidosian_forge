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
@pytest.mark.slow
@pytest.mark.parametrize('scale, exp_scale', [[{'logy': True}, {'yaxis': 'log'}], [{'logx': True}, {'xaxis': 'log'}], [{'loglog': True}, {'xaxis': 'log', 'yaxis': 'log'}]])
def test_plot_scales(self, ts, scale, exp_scale):
    ax = _check_plot_works(ts.plot, style='.', **scale)
    _check_ax_scales(ax, **exp_scale)