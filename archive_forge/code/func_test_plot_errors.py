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
def test_plot_errors():
    with pytest.raises(TypeError, match='plot\\(\\) got an unexpected keyword'):
        plt.plot([1, 2, 3], x=1)
    with pytest.raises(ValueError, match='plot\\(\\) with multiple groups'):
        plt.plot([1, 2, 3], [1, 2, 3], [2, 3, 4], [2, 3, 4], label=['1', '2'])
    with pytest.raises(ValueError, match='x and y must have same first'):
        plt.plot([1, 2, 3], [1])
    with pytest.raises(ValueError, match='x and y can be no greater than'):
        plt.plot(np.ones((2, 2, 2)))
    with pytest.raises(ValueError, match='Using arbitrary long args with'):
        plt.plot('a', 'b', 'c', 'd', data={'a': 2})