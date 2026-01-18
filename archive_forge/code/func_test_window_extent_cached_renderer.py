import collections
import platform
from unittest import mock
import warnings
import numpy as np
from numpy.testing import assert_allclose
import pytest
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple
import matplotlib.legend as mlegend
from matplotlib import _api, rc_context
from matplotlib.font_manager import FontProperties
def test_window_extent_cached_renderer():
    fig, ax = plt.subplots(dpi=100)
    ax.plot(range(10), label='Aardvark')
    leg = ax.legend()
    leg2 = fig.legend()
    fig.canvas.draw()
    leg.get_window_extent()
    leg2.get_window_extent()