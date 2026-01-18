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
def test_plot_series_bar_ax(self):
    ax = _check_plot_works(Series(np.random.default_rng(2).standard_normal(10)).plot.bar, color='black')
    _check_colors([ax.patches[0]], facecolors=['black'])