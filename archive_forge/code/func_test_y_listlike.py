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
@pytest.mark.parametrize('x,y,lbl,colors', [('A', ['B'], ['b'], ['red']), ('A', ['B', 'C'], ['b', 'c'], ['red', 'blue']), (0, [1, 2], ['bokeh', 'cython'], ['green', 'yellow'])])
def test_y_listlike(self, x, y, lbl, colors):
    df = DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
    _check_plot_works(df.plot, x='A', y=y, label=lbl)
    ax = df.plot(x=x, y=y, label=lbl, color=colors)
    assert len(ax.lines) == len(y)
    _check_colors(ax.get_lines(), linecolors=colors)