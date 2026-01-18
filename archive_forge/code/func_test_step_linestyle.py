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
@image_comparison(['step_linestyle', 'step_linestyle'], remove_text=True, tol=0.2)
def test_step_linestyle():
    x = y = np.arange(10)
    fig, ax_lst = plt.subplots(2, 2)
    ax_lst = ax_lst.flatten()
    ln_styles = ['-', '--', '-.', ':']
    for ax, ls in zip(ax_lst, ln_styles):
        ax.step(x, y, lw=5, linestyle=ls, where='pre')
        ax.step(x, y + 1, lw=5, linestyle=ls, where='mid')
        ax.step(x, y + 2, lw=5, linestyle=ls, where='post')
        ax.set_xlim([-1, 5])
        ax.set_ylim([-1, 7])
    data = {'X': x, 'Y0': y, 'Y1': y + 1, 'Y2': y + 2}
    fig, ax_lst = plt.subplots(2, 2)
    ax_lst = ax_lst.flatten()
    ln_styles = ['-', '--', '-.', ':']
    for ax, ls in zip(ax_lst, ln_styles):
        ax.step('X', 'Y0', lw=5, linestyle=ls, where='pre', data=data)
        ax.step('X', 'Y1', lw=5, linestyle=ls, where='mid', data=data)
        ax.step('X', 'Y2', lw=5, linestyle=ls, where='post', data=data)
        ax.set_xlim([-1, 5])
        ax.set_ylim([-1, 7])