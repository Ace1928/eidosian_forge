import functools
import itertools
import platform
import pytest
from mpl_toolkits.mplot3d import Axes3D, axes3d, proj3d, art3d
import matplotlib as mpl
from matplotlib.backend_bases import (MouseButton, MouseEvent,
from matplotlib import cm
from matplotlib import colors as mcolors, patches as mpatch
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.testing.widgets import mock_event
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from matplotlib.text import Text
import matplotlib.pyplot as plt
import numpy as np
def test_proj_transform():
    M = _test_proj_make_M()
    invM = np.linalg.inv(M)
    xs = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0]) * 300.0
    ys = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0]) * 300.0
    zs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) * 300.0
    txs, tys, tzs = proj3d.proj_transform(xs, ys, zs, M)
    ixs, iys, izs = proj3d.inv_transform(txs, tys, tzs, invM)
    np.testing.assert_almost_equal(ixs, xs)
    np.testing.assert_almost_equal(iys, ys)
    np.testing.assert_almost_equal(izs, zs)