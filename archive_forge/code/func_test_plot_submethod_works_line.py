import numpy as np
import pytest
from pandas import (
from pandas.tests.plotting.common import (
def test_plot_submethod_works_line(self):
    df = DataFrame({'x': [1, 2, 3, 4, 5], 'y': [1, 2, 3, 2, 1], 'z': list('ababa')})
    df.groupby('z')['x'].plot.line()