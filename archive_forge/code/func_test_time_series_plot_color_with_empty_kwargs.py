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
def test_time_series_plot_color_with_empty_kwargs(self):
    import matplotlib as mpl
    def_colors = _unpack_cycler(mpl.rcParams)
    index = date_range('1/1/2000', periods=12)
    s = Series(np.arange(1, 13), index=index)
    ncolors = 3
    _, ax = mpl.pyplot.subplots()
    for i in range(ncolors):
        ax = s.plot(ax=ax)
    _check_colors(ax.get_lines(), linecolors=def_colors[:ncolors])