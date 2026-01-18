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
def test_pcolorargs():
    n = 12
    x = np.linspace(-1.5, 1.5, n)
    y = np.linspace(-1.5, 1.5, n * 2)
    X, Y = np.meshgrid(x, y)
    Z = np.hypot(X, Y) / 5
    _, ax = plt.subplots()
    with pytest.raises(TypeError):
        ax.pcolormesh(y, x, Z)
    with pytest.raises(TypeError):
        ax.pcolormesh(X, Y, Z.T)
    with pytest.raises(TypeError):
        ax.pcolormesh(x, y, Z[:-1, :-1], shading='gouraud')
    with pytest.raises(TypeError):
        ax.pcolormesh(X, Y, Z[:-1, :-1], shading='gouraud')
    x[0] = np.nan
    with pytest.raises(ValueError):
        ax.pcolormesh(x, y, Z[:-1, :-1])
    with np.errstate(invalid='ignore'):
        x = np.ma.array(x, mask=x < 0)
    with pytest.raises(ValueError):
        ax.pcolormesh(x, y, Z[:-1, :-1])
    x = [359, 0, 1]
    y = [-10, 10]
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    with pytest.warns(UserWarning, match='are not monotonically increasing or decreasing'):
        ax.pcolormesh(X, Y, Z, shading='auto')