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
def test_pie_series_no_label(self):
    series = Series(np.random.default_rng(2).integers(1, 5), index=['a', 'b', 'c', 'd', 'e'], name='YLABEL')
    ax = _check_plot_works(series.plot.pie, labels=None)
    _check_text_labels(ax.texts, [''] * 5)