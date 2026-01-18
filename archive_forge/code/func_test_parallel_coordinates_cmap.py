import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.slow
def test_parallel_coordinates_cmap(self, iris):
    from matplotlib import cm
    from pandas.plotting import parallel_coordinates
    df = iris
    ax = _check_plot_works(parallel_coordinates, frame=df, class_column='Name', colormap=cm.jet)
    cmaps = [cm.jet(n) for n in np.linspace(0, 1, df['Name'].nunique())]
    _check_colors(ax.get_lines()[:10], linecolors=cmaps, mapping=df['Name'][:10])