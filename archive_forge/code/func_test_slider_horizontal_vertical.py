import functools
import io
from unittest import mock
import matplotlib as mpl
from matplotlib.backend_bases import MouseEvent
import matplotlib.colors as mcolors
import matplotlib.widgets as widgets
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing.widgets import (click_and_drag, do_event, get_ax,
import numpy as np
from numpy.testing import assert_allclose
import pytest
def test_slider_horizontal_vertical():
    fig, ax = plt.subplots()
    slider = widgets.Slider(ax=ax, label='', valmin=0, valmax=24, valinit=12, orientation='horizontal')
    slider.set_val(10)
    assert slider.val == 10
    box = slider.poly.get_extents().transformed(ax.transAxes.inverted())
    assert_allclose(box.bounds, [0, 0.25, 10 / 24, 0.5])
    fig, ax = plt.subplots()
    slider = widgets.Slider(ax=ax, label='', valmin=0, valmax=24, valinit=12, orientation='vertical')
    slider.set_val(10)
    assert slider.val == 10
    box = slider.poly.get_extents().transformed(ax.transAxes.inverted())
    assert_allclose(box.bounds, [0.25, 0, 0.5, 10 / 24])