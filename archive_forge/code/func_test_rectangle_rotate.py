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
@pytest.mark.parametrize('selector_class', [widgets.RectangleSelector, widgets.EllipseSelector])
def test_rectangle_rotate(ax, selector_class):
    tool = selector_class(ax, onselect=noop, interactive=True)
    click_and_drag(tool, start=(100, 100), end=(130, 140))
    assert tool.extents == (100, 130, 100, 140)
    assert len(tool._state) == 0
    do_event(tool, 'on_key_press', key='r')
    assert tool._state == {'rotate'}
    assert len(tool._state) == 1
    click_and_drag(tool, start=(130, 140), end=(120, 145))
    do_event(tool, 'on_key_press', key='r')
    assert len(tool._state) == 0
    assert tool.extents == (100, 130, 100, 140)
    assert_allclose(tool.rotation, 25.56, atol=0.01)
    tool.rotation = 45
    assert tool.rotation == 45
    assert_allclose(tool.corners, np.array([[118.53, 139.75, 111.46, 90.25], [95.25, 116.46, 144.75, 123.54]]), atol=0.01)
    click_and_drag(tool, start=(110, 145), end=(110, 160))
    assert_allclose(tool.extents, (100, 139.75, 100, 151.82), atol=0.01)
    if selector_class == widgets.RectangleSelector:
        with pytest.raises(ValueError):
            tool._selection_artist.rotation_point = 'unvalid_value'