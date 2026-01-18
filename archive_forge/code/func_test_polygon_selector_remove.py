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
@pytest.mark.parametrize('idx', [1, 2, 3])
@pytest.mark.parametrize('draw_bounding_box', [False, True])
def test_polygon_selector_remove(idx, draw_bounding_box):
    verts = [(50, 50), (150, 50), (50, 150)]
    event_sequence = [polygon_place_vertex(*verts[0]), polygon_place_vertex(*verts[1]), polygon_place_vertex(*verts[2]), polygon_place_vertex(*verts[0])]
    event_sequence.insert(idx, polygon_place_vertex(200, 200))
    event_sequence.append(polygon_remove_vertex(200, 200))
    event_sequence = sum(event_sequence, [])
    check_polygon_selector(event_sequence, verts, 2, draw_bounding_box=draw_bounding_box)