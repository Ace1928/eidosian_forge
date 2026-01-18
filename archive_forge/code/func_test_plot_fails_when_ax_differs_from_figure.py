import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_plot_fails_when_ax_differs_from_figure(self, ts):
    from pylab import figure
    fig1 = figure()
    fig2 = figure()
    ax1 = fig1.add_subplot(111)
    msg = 'passed axis not bound to passed figure'
    with pytest.raises(AssertionError, match=msg):
        ts.hist(ax=ax1, figure=fig2)