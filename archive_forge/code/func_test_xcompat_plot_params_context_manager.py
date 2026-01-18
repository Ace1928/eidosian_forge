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
def test_xcompat_plot_params_context_manager(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
    with plotting.plot_params.use('x_compat', True):
        ax = df.plot()
        lines = ax.get_lines()
        assert not isinstance(lines[0].get_xdata(), PeriodIndex)
        _check_ticks_props(ax, xrot=30)