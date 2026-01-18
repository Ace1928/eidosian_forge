import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_radviz_colors_handles(self):
    from pandas.plotting import radviz
    colors = [[0.0, 0.0, 1.0, 1.0], [0.0, 0.5, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]]
    df = DataFrame({'A': [1, 2, 3], 'B': [2, 1, 3], 'C': [3, 2, 1], 'Name': ['b', 'g', 'r']})
    ax = radviz(df, 'Name', color=colors)
    handles, _ = ax.get_legend_handles_labels()
    _check_colors(handles, facecolors=colors)