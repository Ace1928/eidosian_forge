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
def test_quadmesh_set_array():
    x = np.arange(4)
    y = np.arange(4)
    z = np.arange(9).reshape((3, 3))
    fig, ax = plt.subplots()
    coll = ax.pcolormesh(x, y, np.ones(z.shape))
    coll.set_array(z)
    fig.canvas.draw()
    assert np.array_equal(coll.get_array(), z)
    coll.set_array(np.ones(9))
    fig.canvas.draw()
    assert np.array_equal(coll.get_array(), np.ones(9))
    z = np.arange(16).reshape((4, 4))
    fig, ax = plt.subplots()
    coll = ax.pcolormesh(x, y, np.ones(z.shape), shading='gouraud')
    coll.set_array(z)
    fig.canvas.draw()
    assert np.array_equal(coll.get_array(), z)
    coll.set_array(np.ones(16))
    fig.canvas.draw()
    assert np.array_equal(coll.get_array(), np.ones(16))