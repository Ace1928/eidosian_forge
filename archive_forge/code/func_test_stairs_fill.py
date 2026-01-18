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
@check_figures_equal(extensions=['png'])
def test_stairs_fill(fig_test, fig_ref):
    h, bins = ([1, 2, 3, 4, 2], [0, 1, 2, 3, 4, 5])
    bs = -2
    test_axes = fig_test.subplots(2, 2).flatten()
    test_axes[0].stairs(h, bins, fill=True)
    test_axes[1].stairs(h, bins, orientation='horizontal', fill=True)
    test_axes[2].stairs(h, bins, baseline=bs, fill=True)
    test_axes[3].stairs(h, bins, baseline=bs, orientation='horizontal', fill=True)
    ref_axes = fig_ref.subplots(2, 2).flatten()
    ref_axes[0].fill_between(bins, np.append(h, h[-1]), step='post', lw=0)
    ref_axes[0].set_ylim(0, None)
    ref_axes[1].fill_betweenx(bins, np.append(h, h[-1]), step='post', lw=0)
    ref_axes[1].set_xlim(0, None)
    ref_axes[2].fill_between(bins, np.append(h, h[-1]), np.ones(len(h) + 1) * bs, step='post', lw=0)
    ref_axes[2].set_ylim(bs, None)
    ref_axes[3].fill_betweenx(bins, np.append(h, h[-1]), np.ones(len(h) + 1) * bs, step='post', lw=0)
    ref_axes[3].set_xlim(bs, None)