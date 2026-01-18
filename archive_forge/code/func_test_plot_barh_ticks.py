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
def test_plot_barh_ticks(self):
    df = DataFrame({'a': [0, 1], 'b': [1, 0]})
    ax = _check_plot_works(df.plot.barh)
    _check_ticks_props(ax, yrot=0)
    ax = df.plot.barh(rot=55, fontsize=11)
    _check_ticks_props(ax, yrot=55, ylabelsize=11, xlabelsize=11)