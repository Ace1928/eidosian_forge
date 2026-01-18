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
def test_plot_xy_int_cols(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=5, freq='B'))
    df.columns = np.arange(1, len(df.columns) + 1)
    _check_data(df.plot(x=1, y=2), df.set_index(1)[2].plot())
    _check_data(df.plot(x=1), df.set_index(1).plot())
    _check_data(df.plot(y=1), df[1].plot())