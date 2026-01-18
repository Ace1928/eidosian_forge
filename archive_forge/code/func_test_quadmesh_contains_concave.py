from datetime import datetime
import io
import itertools
import re
from types import SimpleNamespace
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
from matplotlib.collections import (Collection, LineCollection,
from matplotlib.testing.decorators import check_figures_equal, image_comparison
def test_quadmesh_contains_concave():
    x = [[0, -1], [1, 0]]
    y = [[0, 1], [1, -1]]
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(x, y, [[0]])
    fig.draw_without_rendering()
    points = [(-0.5, 0.25, True), (0, 0.25, False), (0.5, 0.25, True), (0, -0.25, True)]
    for point in points:
        xdata, ydata, expected = point
        x, y = mesh.get_transform().transform((xdata, ydata))
        mouse_event = SimpleNamespace(xdata=xdata, ydata=ydata, x=x, y=y)
        found, indices = mesh.contains(mouse_event)
        assert found is expected