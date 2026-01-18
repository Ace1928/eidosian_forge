import functools
import itertools
import platform
import pytest
from mpl_toolkits.mplot3d import Axes3D, axes3d, proj3d, art3d
import matplotlib as mpl
from matplotlib.backend_bases import (MouseButton, MouseEvent,
from matplotlib import cm
from matplotlib import colors as mcolors, patches as mpatch
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.testing.widgets import mock_event
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from matplotlib.text import Text
import matplotlib.pyplot as plt
import numpy as np
@check_figures_equal(extensions=['png'])
def test_ticklabel_format(fig_test, fig_ref):
    axs = fig_test.subplots(4, 5, subplot_kw={'projection': '3d'})
    for ax in axs.flat:
        ax.set_xlim(10000000.0, 10000000.0 + 10)
    for row, name in zip(axs, ['x', 'y', 'z', 'both']):
        row[0].ticklabel_format(axis=name, style='plain')
        row[1].ticklabel_format(axis=name, scilimits=(-2, 2))
        row[2].ticklabel_format(axis=name, useOffset=not mpl.rcParams['axes.formatter.useoffset'])
        row[3].ticklabel_format(axis=name, useLocale=not mpl.rcParams['axes.formatter.use_locale'])
        row[4].ticklabel_format(axis=name, useMathText=not mpl.rcParams['axes.formatter.use_mathtext'])

    def get_formatters(ax, names):
        return [getattr(ax, name).get_major_formatter() for name in names]
    axs = fig_ref.subplots(4, 5, subplot_kw={'projection': '3d'})
    for ax in axs.flat:
        ax.set_xlim(10000000.0, 10000000.0 + 10)
    for row, names in zip(axs, [['xaxis'], ['yaxis'], ['zaxis'], ['xaxis', 'yaxis', 'zaxis']]):
        for fmt in get_formatters(row[0], names):
            fmt.set_scientific(False)
        for fmt in get_formatters(row[1], names):
            fmt.set_powerlimits((-2, 2))
        for fmt in get_formatters(row[2], names):
            fmt.set_useOffset(not mpl.rcParams['axes.formatter.useoffset'])
        for fmt in get_formatters(row[3], names):
            fmt.set_useLocale(not mpl.rcParams['axes.formatter.use_locale'])
        for fmt in get_formatters(row[4], names):
            fmt.set_useMathText(not mpl.rcParams['axes.formatter.use_mathtext'])