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
def test_set_zlim():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    assert ax.get_zlim() == (0, 1)
    ax.set_zlim(zmax=2)
    assert ax.get_zlim() == (0, 2)
    ax.set_zlim(zmin=1)
    assert ax.get_zlim() == (1, 2)
    with pytest.raises(TypeError, match="Cannot pass both 'bottom' and 'zmin'"):
        ax.set_zlim(bottom=0, zmin=1)
    with pytest.raises(TypeError, match="Cannot pass both 'top' and 'zmax'"):
        ax.set_zlim(top=0, zmax=1)