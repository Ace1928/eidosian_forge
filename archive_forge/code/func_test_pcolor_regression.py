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
def test_pcolor_regression(pd):
    from pandas.plotting import register_matplotlib_converters, deregister_matplotlib_converters
    fig = plt.figure()
    ax = fig.add_subplot(111)
    times = [datetime.datetime(2021, 1, 1)]
    while len(times) < 7:
        times.append(times[-1] + datetime.timedelta(seconds=120))
    y_vals = np.arange(5)
    time_axis, y_axis = np.meshgrid(times, y_vals)
    shape = (len(y_vals) - 1, len(times) - 1)
    z_data = np.arange(shape[0] * shape[1])
    z_data.shape = shape
    try:
        register_matplotlib_converters()
        im = ax.pcolormesh(time_axis, y_axis, z_data)
        fig.canvas.draw()
    finally:
        deregister_matplotlib_converters()