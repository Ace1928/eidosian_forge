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
def test_set_ticks_with_labels(fig_test, fig_ref):
    """
    Test that these two are identical::

        set_xticks(ticks); set_xticklabels(labels, **kwargs)
        set_xticks(ticks, labels, **kwargs)

    """
    ax = fig_ref.subplots()
    ax.set_xticks([1, 2, 4, 6])
    ax.set_xticklabels(['a', 'b', 'c', 'd'], fontweight='bold')
    ax.set_yticks([1, 3, 5])
    ax.set_yticks([2, 4], minor=True)
    ax.set_yticklabels(['A', 'B'], minor=True)
    ax = fig_test.subplots()
    ax.set_xticks([1, 2, 4, 6], ['a', 'b', 'c', 'd'], fontweight='bold')
    ax.set_yticks([1, 3, 5])
    ax.set_yticks([2, 4], ['A', 'B'], minor=True)