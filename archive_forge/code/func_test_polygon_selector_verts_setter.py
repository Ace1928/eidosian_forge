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
@pytest.mark.parametrize('draw_bounding_box', [False, True])
@check_figures_equal(extensions=['png'])
def test_polygon_selector_verts_setter(fig_test, fig_ref, draw_bounding_box):
    verts = [(0.1, 0.4), (0.5, 0.9), (0.3, 0.2)]
    ax_test = fig_test.add_subplot()
    tool_test = widgets.PolygonSelector(ax_test, onselect=noop, draw_bounding_box=draw_bounding_box)
    tool_test.verts = verts
    assert tool_test.verts == verts
    ax_ref = fig_ref.add_subplot()
    tool_ref = widgets.PolygonSelector(ax_ref, onselect=noop, draw_bounding_box=draw_bounding_box)
    event_sequence = [*polygon_place_vertex(*verts[0]), *polygon_place_vertex(*verts[1]), *polygon_place_vertex(*verts[2]), *polygon_place_vertex(*verts[0])]
    for etype, event_args in event_sequence:
        do_event(tool_ref, etype, **event_args)