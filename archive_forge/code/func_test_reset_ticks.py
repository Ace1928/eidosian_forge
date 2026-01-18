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
def test_reset_ticks(fig_test, fig_ref):
    for fig in [fig_ref, fig_test]:
        ax = fig.add_subplot()
        ax.grid(True)
        ax.tick_params(direction='in', length=10, width=5, color='C0', pad=12, labelsize=14, labelcolor='C1', labelrotation=45, grid_color='C2', grid_alpha=0.8, grid_linewidth=3, grid_linestyle='--')
        fig.draw_without_rendering()
    for ax in fig_test.axes:
        ax.xaxis.reset_ticks()
        ax.yaxis.reset_ticks()