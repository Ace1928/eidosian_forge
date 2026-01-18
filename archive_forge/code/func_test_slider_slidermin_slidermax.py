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
def test_slider_slidermin_slidermax():
    fig, ax = plt.subplots()
    slider_ = widgets.Slider(ax=ax, label='', valmin=0.0, valmax=24.0, valinit=5.0)
    slider = widgets.Slider(ax=ax, label='', valmin=0.0, valmax=24.0, valinit=1.0, slidermin=slider_)
    assert slider.val == slider_.val
    slider = widgets.Slider(ax=ax, label='', valmin=0.0, valmax=24.0, valinit=10.0, slidermax=slider_)
    assert slider.val == slider_.val