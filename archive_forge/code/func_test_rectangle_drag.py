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
@pytest.mark.parametrize('drag_from_anywhere, new_center', [[True, (60, 75)], [False, (30, 20)]])
def test_rectangle_drag(ax, drag_from_anywhere, new_center):
    tool = widgets.RectangleSelector(ax, onselect=noop, interactive=True, drag_from_anywhere=drag_from_anywhere)
    click_and_drag(tool, start=(0, 10), end=(100, 120))
    assert tool.center == (50, 65)
    click_and_drag(tool, start=(25, 15), end=(35, 25))
    assert tool.center == new_center
    click_and_drag(tool, start=(175, 185), end=(185, 195))
    assert tool.center == (180, 190)