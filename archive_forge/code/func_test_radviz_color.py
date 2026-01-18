import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.parametrize('color', [('#556270', '#4ECDC4', '#C7F464'), ['dodgerblue', 'aquamarine', 'seagreen']])
def test_radviz_color(self, iris, color):
    from pandas.plotting import radviz
    df = iris
    ax = _check_plot_works(radviz, frame=df, class_column='Name', color=color)
    patches = [p for p in ax.patches[:20] if p.get_label() != '']
    _check_colors(patches[:10], facecolors=color, mapping=df['Name'][:10])