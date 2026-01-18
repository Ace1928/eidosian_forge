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
def test_scatter_no_invalid_color(self, fig_test, fig_ref):
    ax = fig_test.subplots()
    cmap = mpl.colormaps['viridis'].resampled(16)
    cmap.set_bad('k', 1)
    ax.scatter(range(4), range(4), c=[1, np.nan, 2, np.nan], s=[1, 2, 3, 4], cmap=cmap, plotnonfinite=False)
    ax = fig_ref.subplots()
    ax.scatter([0, 2], [0, 2], c=[1, 2], s=[1, 3], cmap=cmap)