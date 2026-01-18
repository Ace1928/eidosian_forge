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
def test_relim_visible_only():
    x1 = (0.0, 10.0)
    y1 = (0.0, 10.0)
    x2 = (-10.0, 20.0)
    y2 = (-10.0, 30.0)
    fig = matplotlib.figure.Figure()
    ax = fig.add_subplot()
    ax.plot(x1, y1)
    assert ax.get_xlim() == x1
    assert ax.get_ylim() == y1
    line, = ax.plot(x2, y2)
    assert ax.get_xlim() == x2
    assert ax.get_ylim() == y2
    line.set_visible(False)
    assert ax.get_xlim() == x2
    assert ax.get_ylim() == y2
    ax.relim(visible_only=True)
    ax.autoscale_view()
    assert ax.get_xlim() == x1
    assert ax.get_ylim() == y1