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
def test_quadmesh_get_coordinates(pcfunc):
    x = [0, 1, 2]
    y = [2, 4, 6]
    z = np.ones(shape=(2, 2))
    xx, yy = np.meshgrid(x, y)
    coll = getattr(plt, pcfunc)(xx, yy, z)
    coords = np.stack([xx.T, yy.T]).T
    assert_array_equal(coll.get_coordinates(), coords)