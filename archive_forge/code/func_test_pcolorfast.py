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
@pytest.mark.parametrize('xy, cls', [((), mpl.image.AxesImage), (((3, 7), (2, 6)), mpl.image.AxesImage), ((range(5), range(4)), mpl.image.AxesImage), (([1, 2, 4, 8, 16], [0, 1, 2, 3]), mpl.image.PcolorImage), ((np.random.random((4, 5)), np.random.random((4, 5))), mpl.collections.QuadMesh)])
@pytest.mark.parametrize('data', [np.arange(12).reshape((3, 4)), np.random.rand(3, 4, 3)])
def test_pcolorfast(xy, data, cls):
    fig, ax = plt.subplots()
    assert type(ax.pcolorfast(*xy, data)) == cls