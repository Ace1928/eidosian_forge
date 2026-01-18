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
@pytest.mark.parametrize('kwargs, expected_edgecolors', [(dict(), None), (dict(c='b'), None), (dict(edgecolors='r'), 'r'), (dict(edgecolors=['r', 'g']), ['r', 'g']), (dict(edgecolor='r'), 'r'), (dict(edgecolors='face'), 'face'), (dict(edgecolors='none'), 'none'), (dict(edgecolor='r', edgecolors='g'), 'r'), (dict(c='b', edgecolor='r', edgecolors='g'), 'r'), (dict(color='r'), 'r'), (dict(color='r', edgecolor='g'), 'g')])
def test_parse_scatter_color_args_edgecolors(kwargs, expected_edgecolors):

    def get_next_color():
        return 'blue'
    c = kwargs.pop('c', None)
    edgecolors = kwargs.pop('edgecolors', None)
    _, _, result_edgecolors = mpl.axes.Axes._parse_scatter_color_args(c, edgecolors, kwargs, xsize=2, get_next_color_func=get_next_color)
    assert result_edgecolors == expected_edgecolors