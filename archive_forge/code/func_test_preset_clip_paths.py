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
@image_comparison(['preset_clip_paths.png'], remove_text=True, style='mpl20')
def test_preset_clip_paths():
    fig, ax = plt.subplots()
    poly = mpl.patches.Polygon([[1, 0], [0, 1], [-1, 0], [0, -1]], facecolor='#ddffdd', edgecolor='#00ff00', linewidth=2, alpha=0.5)
    ax.add_patch(poly)
    line = mpl.lines.Line2D((-1, 1), (0.5, 0.5), clip_on=True, clip_path=poly)
    line.set_path_effects([patheffects.withTickedStroke()])
    ax.add_artist(line)
    line = mpl.lines.Line2D((-1, 1), (-0.5, -0.5), color='r', clip_on=True, clip_path=poly)
    ax.add_artist(line)
    poly2 = mpl.patches.Polygon([[-1, 1], [0, 1], [0, -0.25]], facecolor='#beefc0', alpha=0.3, edgecolor='#faded0', linewidth=2, clip_on=True, clip_path=poly)
    ax.add_artist(poly2)
    ax.annotate('Annotation', (-0.75, -0.75), xytext=(0.1, 0.75), arrowprops={'color': 'k'}, clip_on=True, clip_path=poly)
    poly3 = mpl.patches.Polygon([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]], facecolor='g', edgecolor='y', linewidth=2, alpha=0.3, clip_on=True, clip_path=poly)
    fig.add_artist(poly3, clip=True)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)