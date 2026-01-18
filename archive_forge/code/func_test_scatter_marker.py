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
@image_comparison(['scatter_marker.png'], remove_text=True)
def test_scatter_marker(self):
    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3)
    ax0.scatter([3, 4, 2, 6], [2, 5, 2, 3], c=[(1, 0, 0), 'y', 'b', 'lime'], s=[60, 50, 40, 30], edgecolors=['k', 'r', 'g', 'b'], marker='s')
    ax1.scatter([3, 4, 2, 6], [2, 5, 2, 3], c=[(1, 0, 0), 'y', 'b', 'lime'], s=[60, 50, 40, 30], edgecolors=['k', 'r', 'g', 'b'], marker=mmarkers.MarkerStyle('o', fillstyle='top'))
    rx, ry = (3, 1)
    area = rx * ry * np.pi
    theta = np.linspace(0, 2 * np.pi, 21)
    verts = np.column_stack([np.cos(theta) * rx / area, np.sin(theta) * ry / area])
    ax2.scatter([3, 4, 2, 6], [2, 5, 2, 3], c=[(1, 0, 0), 'y', 'b', 'lime'], s=[60, 50, 40, 30], edgecolors=['k', 'r', 'g', 'b'], marker=verts)