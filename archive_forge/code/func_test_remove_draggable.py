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
def test_remove_draggable():
    fig, ax = plt.subplots()
    an = ax.annotate('foo', (0.5, 0.5))
    an.draggable(True)
    an.remove()
    MouseEvent('button_release_event', fig.canvas, 1, 1)._process()