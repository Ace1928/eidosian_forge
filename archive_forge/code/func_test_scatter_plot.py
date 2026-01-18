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
@image_comparison(['scatter'], style='mpl20', remove_text=True)
def test_scatter_plot(self):
    data = {'x': np.array([3, 4, 2, 6]), 'y': np.array([2, 5, 2, 3]), 'c': ['r', 'y', 'b', 'lime'], 's': [24, 15, 19, 29], 'c2': ['0.5', '0.6', '0.7', '0.8']}
    fig, ax = plt.subplots()
    ax.scatter(data['x'] - 1.0, data['y'] - 1.0, c=data['c'], s=data['s'])
    ax.scatter(data['x'] + 1.0, data['y'] + 1.0, c=data['c2'], s=data['s'])
    ax.scatter('x', 'y', c='c', s='s', data=data)