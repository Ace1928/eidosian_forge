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
@image_comparison(['pie_shadow.png'], style='mpl20', tol=0.002)
def test_pie_shadow():
    sizes = [15, 30, 45, 10]
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    explode = (0, 0.1, 0, 0)
    _, axes = plt.subplots(2, 2)
    axes[0][0].pie(sizes, explode=explode, colors=colors, shadow=True, startangle=90, wedgeprops={'linewidth': 0})
    axes[0][1].pie(sizes, explode=explode, colors=colors, shadow=False, startangle=90, wedgeprops={'linewidth': 0})
    axes[1][0].pie(sizes, explode=explode, colors=colors, shadow={'ox': -0.05, 'oy': -0.05, 'shade': 0.9, 'edgecolor': 'none'}, startangle=90, wedgeprops={'linewidth': 0})
    axes[1][1].pie(sizes, explode=explode, colors=colors, shadow={'ox': 0.05, 'linewidth': 2, 'shade': 0.2}, startangle=90, wedgeprops={'linewidth': 0})