import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.patches import (Annulus, Ellipse, Patch, Polygon, Rectangle,
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
from matplotlib import (
import sys
def test_rotate_rect():
    loc = np.asarray([1.0, 2.0])
    width = 2
    height = 3
    angle = 30.0
    rect1 = Rectangle(loc, width, height, angle=angle)
    rect2 = Rectangle(loc, width, height)
    angle_rad = np.pi * angle / 180.0
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
    new_verts = np.inner(rotation_matrix, rect2.get_verts() - loc).T + loc
    assert_almost_equal(rect1.get_verts(), new_verts)