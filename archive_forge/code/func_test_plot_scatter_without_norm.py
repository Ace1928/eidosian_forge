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
def test_plot_scatter_without_norm(self):
    df = DataFrame(np.random.default_rng(2).random((10, 3)) * 100, columns=['a', 'b', 'c'])
    ax = df.plot.scatter(x='a', y='b', c='c')
    plot_norm = ax.collections[0].norm
    color_min_max = (df.c.min(), df.c.max())
    default_norm = mpl.colors.Normalize(*color_min_max)
    for value in df.c:
        assert plot_norm(value) == default_norm(value)