import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import Rectangle
def test_outward_ticks():
    """Test automatic use of tight_layout."""
    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.xaxis.set_tick_params(tickdir='out', length=16, width=3)
    ax.yaxis.set_tick_params(tickdir='out', length=16, width=3)
    ax.xaxis.set_tick_params(tickdir='out', length=32, width=3, tick1On=True, which='minor')
    ax.yaxis.set_tick_params(tickdir='out', length=32, width=3, tick1On=True, which='minor')
    ax.xaxis.set_ticks([0], minor=True)
    ax.yaxis.set_ticks([0], minor=True)
    ax = fig.add_subplot(222)
    ax.xaxis.set_tick_params(tickdir='in', length=32, width=3)
    ax.yaxis.set_tick_params(tickdir='in', length=32, width=3)
    ax = fig.add_subplot(223)
    ax.xaxis.set_tick_params(tickdir='inout', length=32, width=3)
    ax.yaxis.set_tick_params(tickdir='inout', length=32, width=3)
    ax = fig.add_subplot(224)
    ax.xaxis.set_tick_params(tickdir='out', length=32, width=3)
    ax.yaxis.set_tick_params(tickdir='out', length=32, width=3)
    plt.tight_layout()
    ans = [[[0.091, 0.607], [0.433, 0.933]], [[0.579, 0.607], [0.922, 0.933]], [[0.091, 0.14], [0.433, 0.466]], [[0.579, 0.14], [0.922, 0.466]]]
    for nn, ax in enumerate(fig.axes):
        assert_array_equal(np.round(ax.get_position().get_points(), 3), ans[nn])