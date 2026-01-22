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
def _bxp_test_helper(stats_kwargs={}, transform_stats=lambda s: s, bxp_kwargs={}):
    np.random.seed(937)
    logstats = mpl.cbook.boxplot_stats(np.random.lognormal(mean=1.25, sigma=1.0, size=(37, 4)), **stats_kwargs)
    fig, ax = plt.subplots()
    if bxp_kwargs.get('vert', True):
        ax.set_yscale('log')
    else:
        ax.set_xscale('log')
    if not bxp_kwargs.get('patch_artist', False):
        mpl.rcParams['boxplot.boxprops.linewidth'] = mpl.rcParams['lines.linewidth']
    ax.bxp(transform_stats(logstats), **bxp_kwargs)