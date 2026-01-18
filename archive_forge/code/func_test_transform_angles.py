import copy
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
import pytest
from matplotlib import scale
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.transforms import Affine2D, Bbox, TransformedBbox
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_transform_angles():
    t = mtransforms.Affine2D()
    angles = np.array([20, 45, 60])
    points = np.array([[0, 0], [1, 1], [2, 2]])
    new_angles = t.transform_angles(angles, points)
    assert_array_almost_equal(angles, new_angles)
    with pytest.raises(ValueError):
        t.transform_angles(angles, points[0:2, 0:1])
    with pytest.raises(ValueError):
        t.transform_angles(angles, points[0:2, :])