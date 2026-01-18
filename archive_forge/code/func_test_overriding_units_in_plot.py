import warnings
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.category as cat
from matplotlib.testing.decorators import check_figures_equal
@mpl.style.context('default')
@check_figures_equal(extensions=['png'])
def test_overriding_units_in_plot(fig_test, fig_ref):
    from datetime import datetime
    t0 = datetime(2018, 3, 1)
    t1 = datetime(2018, 3, 2)
    t2 = datetime(2018, 3, 3)
    t3 = datetime(2018, 3, 4)
    ax_test = fig_test.subplots()
    ax_ref = fig_ref.subplots()
    for ax, kwargs in zip([ax_test, ax_ref], ({}, dict(xunits=None, yunits=None))):
        ax.plot([t0, t1], ['V1', 'V2'], **kwargs)
        x_units = ax.xaxis.units
        y_units = ax.yaxis.units
        ax.plot([t2, t3], ['V1', 'V2'], **kwargs)
        assert x_units is ax.xaxis.units
        assert y_units is ax.yaxis.units