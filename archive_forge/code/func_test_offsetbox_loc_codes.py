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
def test_offsetbox_loc_codes():
    codes = {'upper right': 1, 'upper left': 2, 'lower left': 3, 'lower right': 4, 'right': 5, 'center left': 6, 'center right': 7, 'lower center': 8, 'upper center': 9, 'center': 10}
    fig, ax = plt.subplots()
    da = DrawingArea(100, 100)
    for code in codes:
        anchored_box = AnchoredOffsetbox(loc=code, child=da)
        ax.add_artist(anchored_box)
    fig.canvas.draw()