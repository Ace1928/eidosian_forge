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
def test_pie_textprops():
    data = [23, 34, 45]
    labels = ['Long name 1', 'Long name 2', 'Long name 3']
    textprops = dict(horizontalalignment='center', verticalalignment='top', rotation=90, rotation_mode='anchor', size=12, color='red')
    _, texts, autopct = plt.gca().pie(data, labels=labels, autopct='%.2f', textprops=textprops)
    for labels in [texts, autopct]:
        for tx in labels:
            assert tx.get_ha() == textprops['horizontalalignment']
            assert tx.get_va() == textprops['verticalalignment']
            assert tx.get_rotation() == textprops['rotation']
            assert tx.get_rotation_mode() == textprops['rotation_mode']
            assert tx.get_size() == textprops['size']
            assert tx.get_color() == textprops['color']