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
def test_polygon_selector_box(ax):
    ax.set(xlim=(-10, 50), ylim=(-10, 50))
    verts = [(20, 0), (0, 20), (20, 40), (40, 20)]
    event_sequence = [*polygon_place_vertex(*verts[0]), *polygon_place_vertex(*verts[1]), *polygon_place_vertex(*verts[2]), *polygon_place_vertex(*verts[3]), *polygon_place_vertex(*verts[0])]
    tool = widgets.PolygonSelector(ax, onselect=noop, draw_bounding_box=True)
    for etype, event_args in event_sequence:
        do_event(tool, etype, **event_args)
    t = ax.transData
    canvas = ax.figure.canvas
    MouseEvent('button_press_event', canvas, *t.transform((40, 40)), 1)._process()
    MouseEvent('motion_notify_event', canvas, *t.transform((20, 20)))._process()
    MouseEvent('button_release_event', canvas, *t.transform((20, 20)), 1)._process()
    np.testing.assert_allclose(tool.verts, [(10, 0), (0, 10), (10, 20), (20, 10)])
    MouseEvent('button_press_event', canvas, *t.transform((10, 10)), 1)._process()
    MouseEvent('motion_notify_event', canvas, *t.transform((30, 30)))._process()
    MouseEvent('button_release_event', canvas, *t.transform((30, 30)), 1)._process()
    np.testing.assert_allclose(tool.verts, [(30, 20), (20, 30), (30, 40), (40, 30)])
    np.testing.assert_allclose(tool._box.extents, (20.0, 40.0, 20.0, 40.0))
    MouseEvent('button_press_event', canvas, *t.transform((30, 20)), 3)._process()
    MouseEvent('button_release_event', canvas, *t.transform((30, 20)), 3)._process()
    np.testing.assert_allclose(tool.verts, [(20, 30), (30, 40), (40, 30)])
    np.testing.assert_allclose(tool._box.extents, (20.0, 40.0, 30.0, 40.0))