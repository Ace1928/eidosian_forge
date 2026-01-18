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
@image_comparison(['pcolormesh_alpha'], extensions=['png', 'pdf'], remove_text=True)
def test_pcolormesh_alpha():
    plt.rcParams['pcolormesh.snap'] = False
    n = 12
    X, Y = np.meshgrid(np.linspace(-1.5, 1.5, n), np.linspace(-1.5, 1.5, n * 2))
    Qx = X
    Qy = Y + np.sin(X)
    Z = np.hypot(X, Y) / 5
    Z = (Z - Z.min()) / np.ptp(Z)
    vir = mpl.colormaps['viridis'].resampled(16)
    colors = vir(np.arange(16))
    colors[:, 3] = 0.5 + 0.5 * np.sin(np.arange(16))
    cmap = mcolors.ListedColormap(colors)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    for ax in (ax1, ax2, ax3, ax4):
        ax.add_patch(mpatches.Rectangle((0, -1.5), 1.5, 3, facecolor=[0.7, 0.1, 0.1, 0.5], zorder=0))
    ax1.pcolormesh(Qx, Qy, Z[:-1, :-1], cmap=vir, alpha=0.4, shading='flat', zorder=1)
    ax2.pcolormesh(Qx, Qy, Z, cmap=vir, alpha=0.4, shading='gouraud', zorder=1)
    ax3.pcolormesh(Qx, Qy, Z[:-1, :-1], cmap=cmap, shading='flat', zorder=1)
    ax4.pcolormesh(Qx, Qy, Z, cmap=cmap, shading='gouraud', zorder=1)