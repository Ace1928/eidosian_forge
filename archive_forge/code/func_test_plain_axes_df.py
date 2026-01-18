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
def test_plain_axes_df(self):
    df = DataFrame({'a': np.random.default_rng(2).standard_normal(8), 'b': np.random.default_rng(2).standard_normal(8)})
    fig = mpl.pyplot.figure()
    ax = fig.add_axes((0, 0, 1, 1))
    df.plot(kind='scatter', ax=ax, x='a', y='b', c='a', cmap='hsv')