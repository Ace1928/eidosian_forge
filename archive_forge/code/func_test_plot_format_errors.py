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
@pytest.mark.parametrize('fmt, match', (('f', "'f' is not a valid format string \\(unrecognized character 'f'\\)"), ('o+', "'o\\+' is not a valid format string \\(two marker symbols\\)"), (':-', "':-' is not a valid format string \\(two linestyle symbols\\)"), ('rk', "'rk' is not a valid format string \\(two color symbols\\)"), (':o-r', "':o-r' is not a valid format string \\(two linestyle symbols\\)")))
@pytest.mark.parametrize('data', [None, {'string': range(3)}])
def test_plot_format_errors(fmt, match, data):
    fig, ax = plt.subplots()
    if data is not None:
        match = match.replace('not', 'neither a data key nor')
    with pytest.raises(ValueError, match='\\A' + match + '\\Z'):
        ax.plot('string', fmt, data=data)