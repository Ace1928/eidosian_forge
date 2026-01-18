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
def test_plot_format():
    fig, ax = plt.subplots()
    line = ax.plot([1, 2, 3], '1.0')
    assert line[0].get_color() == (1.0, 1.0, 1.0, 1.0)
    assert line[0].get_marker() == 'None'
    fig, ax = plt.subplots()
    line = ax.plot([1, 2, 3], '1')
    assert line[0].get_marker() == '1'
    fig, ax = plt.subplots()
    line = ax.plot([1, 2], [1, 2], '1.0', '1')
    fig.canvas.draw()
    assert line[0].get_color() == (1.0, 1.0, 1.0, 1.0)
    assert ax.get_yticklabels()[0].get_text() == '1'
    fig, ax = plt.subplots()
    line = ax.plot([1, 2], [1, 2], '1', '1.0')
    fig.canvas.draw()
    assert line[0].get_marker() == '1'
    assert ax.get_yticklabels()[0].get_text() == '1.0'
    fig, ax = plt.subplots()
    line = ax.plot([1, 2, 3], 'k3')
    assert line[0].get_marker() == '3'
    assert line[0].get_color() == 'k'