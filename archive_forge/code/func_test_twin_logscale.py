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
@pytest.mark.parametrize('twin', ('x', 'y'))
@check_figures_equal(extensions=['png'], tol=0.19)
def test_twin_logscale(fig_test, fig_ref, twin):
    twin_func = f'twin{twin}'
    set_scale = f'set_{twin}scale'
    x = np.arange(1, 100)
    ax_test = fig_test.add_subplot(2, 1, 1)
    ax_twin = getattr(ax_test, twin_func)()
    getattr(ax_test, set_scale)('log')
    ax_twin.plot(x, x)
    ax_test = fig_test.add_subplot(2, 1, 2)
    getattr(ax_test, set_scale)('log')
    ax_twin = getattr(ax_test, twin_func)()
    ax_twin.plot(x, x)
    for i in [1, 2]:
        ax_ref = fig_ref.add_subplot(2, 1, i)
        getattr(ax_ref, set_scale)('log')
        ax_ref.plot(x, x)
        Path = matplotlib.path.Path
        fig_ref.add_artist(matplotlib.patches.PathPatch(Path([[0, 0], [0, 1], [0, 1], [1, 1], [1, 1], [1, 0], [1, 0], [0, 0]], [Path.MOVETO, Path.LINETO] * 4), transform=ax_ref.transAxes, facecolor='none', edgecolor=mpl.rcParams['axes.edgecolor'], linewidth=mpl.rcParams['axes.linewidth'], capstyle='projecting'))
    remove_ticks_and_titles(fig_test)
    remove_ticks_and_titles(fig_ref)