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
@check_figures_equal(extensions=['png'])
def test_radio_buttons(fig_test, fig_ref):
    widgets.RadioButtons(fig_test.subplots(), ['tea', 'coffee'])
    ax = fig_ref.add_subplot(xticks=[], yticks=[])
    ax.scatter([0.15, 0.15], [2 / 3, 1 / 3], transform=ax.transAxes, s=(plt.rcParams['font.size'] / 2) ** 2, c=['C0', 'none'])
    ax.text(0.25, 2 / 3, 'tea', transform=ax.transAxes, va='center')
    ax.text(0.25, 1 / 3, 'coffee', transform=ax.transAxes, va='center')