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
@check_figures_equal(extensions=['png'])
def test_text_3d(fig_test, fig_ref):
    ax = fig_ref.add_subplot(projection='3d')
    txt = Text(0.5, 0.5, 'Foo bar $\\int$')
    art3d.text_2d_to_3d(txt, z=1)
    ax.add_artist(txt)
    assert txt.get_position_3d() == (0.5, 0.5, 1)
    ax = fig_test.add_subplot(projection='3d')
    t3d = art3d.Text3D(0.5, 0.5, 1, 'Foo bar $\\int$')
    ax.add_artist(t3d)
    assert t3d.get_position_3d() == (0.5, 0.5, 1)