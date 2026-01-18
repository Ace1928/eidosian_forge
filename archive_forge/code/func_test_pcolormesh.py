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
@image_comparison(['pcolormesh'], remove_text=True)
def test_pcolormesh():
    plt.rcParams['pcolormesh.snap'] = False
    n = 12
    x = np.linspace(-1.5, 1.5, n)
    y = np.linspace(-1.5, 1.5, n * 2)
    X, Y = np.meshgrid(x, y)
    Qx = np.cos(Y) - np.cos(X)
    Qz = np.sin(Y) + np.sin(X)
    Qx = Qx + 1.1
    Z = np.hypot(X, Y) / 5
    Z = (Z - Z.min()) / np.ptp(Z)
    Zm = ma.masked_where(np.abs(Qz) < 0.5 * np.max(Qz), Z)
    _, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.pcolormesh(Qx, Qz, Zm[:-1, :-1], lw=0.5, edgecolors='k')
    ax2.pcolormesh(Qx, Qz, Zm[:-1, :-1], lw=2, edgecolors=['b', 'w'])
    ax3.pcolormesh(Qx, Qz, Zm, shading='gouraud')