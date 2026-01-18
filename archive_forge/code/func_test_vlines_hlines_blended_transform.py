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
@image_comparison(['vlines_hlines_blended_transform'], extensions=['png'], style='mpl20')
def test_vlines_hlines_blended_transform():
    t = np.arange(5.0, 10.0, 0.1)
    s = np.exp(-t) + np.sin(2 * np.pi * t) + 10
    fig, (hax, vax) = plt.subplots(2, 1, figsize=(6, 6))
    hax.plot(t, s, '^')
    hax.hlines([10, 9], xmin=0, xmax=0.5, transform=hax.get_yaxis_transform(), colors='r')
    vax.plot(t, s, '^')
    vax.vlines([6, 7], ymin=0, ymax=0.15, transform=vax.get_xaxis_transform(), colors='r')