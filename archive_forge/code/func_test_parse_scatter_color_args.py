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
@pytest.mark.parametrize('params, expected_result', [(_params(), _result(c='b', colors=np.array([[0, 0, 1, 1]]))), (_params(c='r'), _result(c='r', colors=np.array([[1, 0, 0, 1]]))), (_params(c='r', colors='b'), _result(c='r', colors=np.array([[1, 0, 0, 1]]))), (_params(color='b'), _result(c='b', colors=np.array([[0, 0, 1, 1]]))), (_params(color=['b', 'g']), _result(c=['b', 'g'], colors=np.array([[0, 0, 1, 1], [0, 0.5, 0, 1]])))])
def test_parse_scatter_color_args(params, expected_result):

    def get_next_color():
        return 'blue'
    c, colors, _edgecolors = mpl.axes.Axes._parse_scatter_color_args(*params, get_next_color_func=get_next_color)
    assert c == expected_result.c
    assert_allclose(colors, expected_result.colors)