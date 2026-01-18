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
def test_slider_reset():
    fig, ax = plt.subplots()
    slider = widgets.Slider(ax=ax, label='', valmin=0, valmax=1, valinit=0.5)
    slider.set_val(0.75)
    slider.reset()
    assert slider.val == 0.5