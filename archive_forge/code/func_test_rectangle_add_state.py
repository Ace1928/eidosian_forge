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
def test_rectangle_add_state(ax):
    tool = widgets.RectangleSelector(ax, onselect=noop, interactive=True)
    click_and_drag(tool, start=(70, 65), end=(125, 130))
    with pytest.raises(ValueError):
        tool.add_state('unsupported_state')
    with pytest.raises(ValueError):
        tool.add_state('clear')
    tool.add_state('move')
    tool.add_state('square')
    tool.add_state('center')