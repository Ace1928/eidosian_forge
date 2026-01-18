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
def test_plot_fails_with_dupe_color_and_style(self):
    x = Series(np.random.default_rng(2).standard_normal(2))
    _, ax = mpl.pyplot.subplots()
    msg = "Cannot pass 'style' string with a color symbol and 'color' keyword argument. Please use one or the other or pass 'style' without a color symbol"
    with pytest.raises(ValueError, match=msg):
        x.plot(style='k--', color='k', ax=ax)