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
def test_plot_scatter_with_c_array(self):
    df = DataFrame({'A': [1, 2], 'B': [3, 4]})
    red_rgba = [1.0, 0.0, 0.0, 1.0]
    green_rgba = [0.0, 1.0, 0.0, 1.0]
    rgba_array = np.array([red_rgba, green_rgba])
    ax = df.plot.scatter(x='A', y='B', c=rgba_array)
    tm.assert_numpy_array_equal(ax.collections[0].get_facecolor(), rgba_array)
    float_array = np.array([0.0, 1.0])
    df.plot.scatter(x='A', y='B', c=float_array, cmap='spring')