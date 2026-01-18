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
@pytest.mark.parametrize('dims,alpha', [(3, 1), (4, 0.5)])
@check_figures_equal(extensions=['png'])
def test_pcolormesh_rgba(fig_test, fig_ref, dims, alpha):
    ax = fig_test.subplots()
    c = np.ones((5, 6, dims), dtype=float) / 2
    ax.pcolormesh(c)
    ax = fig_ref.subplots()
    ax.pcolormesh(c[..., 0], cmap='gray', vmin=0, vmax=1, alpha=alpha)