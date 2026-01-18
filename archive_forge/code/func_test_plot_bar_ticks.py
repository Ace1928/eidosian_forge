from datetime import (
import gc
import itertools
import re
import string
import weakref
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.slow
def test_plot_bar_ticks(self):
    df = DataFrame({'a': [0, 1], 'b': [1, 0]})
    ax = _check_plot_works(df.plot.bar)
    _check_ticks_props(ax, xrot=90)
    ax = df.plot.bar(rot=35, fontsize=10)
    _check_ticks_props(ax, xrot=35, xlabelsize=10, ylabelsize=10)