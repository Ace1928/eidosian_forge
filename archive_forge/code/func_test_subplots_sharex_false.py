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
def test_subplots_sharex_false(self):
    df = DataFrame(np.random.default_rng(2).random((10, 2)))
    df.iloc[5:, 1] = np.nan
    df.iloc[:5, 0] = np.nan
    _, axs = mpl.pyplot.subplots(2, 1)
    df.plot.line(ax=axs, subplots=True, sharex=False)
    expected_ax1 = np.arange(4.5, 10, 0.5)
    expected_ax2 = np.arange(-0.5, 5, 0.5)
    tm.assert_numpy_array_equal(axs[0].get_xticks(), expected_ax1)
    tm.assert_numpy_array_equal(axs[1].get_xticks(), expected_ax2)