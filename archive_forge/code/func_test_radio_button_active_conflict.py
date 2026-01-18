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
def test_radio_button_active_conflict(ax):
    with pytest.warns(UserWarning, match='Both the \\*activecolor\\* parameter'):
        rb = widgets.RadioButtons(ax, ['tea', 'coffee'], activecolor='red', radio_props={'facecolor': 'green'})
    assert mcolors.same_color(rb._buttons.get_facecolor(), ['green', 'none'])