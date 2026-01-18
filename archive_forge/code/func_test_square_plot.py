import contextlib
from collections import namedtuple
import datetime
from decimal import Decimal
from functools import partial
import inspect
import io
from itertools import product
import platform
from types import SimpleNamespace
import dateutil.tz
import numpy as np
from numpy import ma
from cycler import cycler
import pytest
import matplotlib
import matplotlib as mpl
from matplotlib import rc_context, patheffects
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.font_manager as mfont_manager
import matplotlib.markers as mmarkers
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from matplotlib.projections.geo import HammerAxes
from matplotlib.projections.polar import PolarAxes
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import mpl_toolkits.axisartist as AA  # type: ignore
from numpy.testing import (
from matplotlib.testing.decorators import (
def test_square_plot():
    x = np.arange(4)
    y = np.array([1.0, 3.0, 5.0, 7.0])
    fig, ax = plt.subplots()
    ax.plot(x, y, 'mo')
    ax.axis('square')
    xlim, ylim = (ax.get_xlim(), ax.get_ylim())
    assert np.diff(xlim) == np.diff(ylim)
    assert ax.get_aspect() == 1
    assert_array_almost_equal(ax.get_position(original=True).extents, (0.125, 0.1, 0.9, 0.9))
    assert_array_almost_equal(ax.get_position(original=False).extents, (0.2125, 0.1, 0.8125, 0.9))