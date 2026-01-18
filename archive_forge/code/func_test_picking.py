from collections import namedtuple
import io
import numpy as np
from numpy.testing import assert_allclose
import pytest
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.offsetbox import (
@pytest.mark.parametrize('child_type', ['draw', 'image', 'text'])
@pytest.mark.parametrize('boxcoords', ['axes fraction', 'axes pixels', 'axes points', 'data'])
def test_picking(child_type, boxcoords):
    if child_type == 'draw':
        picking_child = DrawingArea(5, 5)
        picking_child.add_artist(mpatches.Rectangle((0, 0), 5, 5, linewidth=0))
    elif child_type == 'image':
        im = np.ones((5, 5))
        im[2, 2] = 0
        picking_child = OffsetImage(im)
    elif child_type == 'text':
        picking_child = TextArea('■', textprops={'fontsize': 5})
    else:
        assert False, f'Unknown picking child type {child_type}'
    fig, ax = plt.subplots()
    ab = AnnotationBbox(picking_child, (0.5, 0.5), boxcoords=boxcoords)
    ab.set_picker(True)
    ax.add_artist(ab)
    calls = []
    fig.canvas.mpl_connect('pick_event', lambda event: calls.append(event))
    if boxcoords == 'axes points':
        x, y = ax.transAxes.transform_point((0, 0))
        x += 0.5 * fig.dpi / 72
        y += 0.5 * fig.dpi / 72
    elif boxcoords == 'axes pixels':
        x, y = ax.transAxes.transform_point((0, 0))
        x += 0.5
        y += 0.5
    else:
        x, y = ax.transAxes.transform_point((0.5, 0.5))
    fig.canvas.draw()
    calls.clear()
    MouseEvent('button_press_event', fig.canvas, x, y, MouseButton.LEFT)._process()
    assert len(calls) == 1 and calls[0].artist == ab
    ax.set_xlim(-1, 0)
    ax.set_ylim(-1, 0)
    fig.canvas.draw()
    calls.clear()
    MouseEvent('button_press_event', fig.canvas, x, y, MouseButton.LEFT)._process()
    assert len(calls) == 0