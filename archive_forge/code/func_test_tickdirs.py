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
def test_tickdirs():
    """
    Switch the tickdirs and make sure the bboxes switch with them
    """
    targets = [[[150.0, 120.0, 930.0, 11.1111], [150.0, 120.0, 11.111, 960.0]], [[150.0, 108.8889, 930.0, 11.111111111111114], [138.889, 120, 11.111, 960.0]], [[150.0, 114.44444444444441, 930.0, 11.111111111111114], [144.44444444444446, 119.999, 11.111, 960.0]]]
    for dnum, dirs in enumerate(['in', 'out', 'inout']):
        with rc_context({'_internal.classic_mode': False}):
            fig, ax = plt.subplots(dpi=200, figsize=(6, 6))
            ax.tick_params(direction=dirs)
            fig.canvas.draw()
            bbaxis, bbspines, bbax, bbtb = color_boxes(fig, ax)
            for nn, num in enumerate([0, 2]):
                targetbb = mtransforms.Bbox.from_bounds(*targets[dnum][nn])
                assert_allclose(bbspines[num].bounds, targetbb.bounds, atol=0.01)