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
def test_transformed_path():
    points = [(0, 0), (1, 0), (1, 1), (0, 1)]
    path = Path(points, closed=True)
    trans = mtransforms.Affine2D()
    trans_path = mtransforms.TransformedPath(path, trans)
    assert_allclose(trans_path.get_fully_transformed_path().vertices, points)
    r2 = 1 / np.sqrt(2)
    trans.rotate(np.pi / 4)
    assert_allclose(trans_path.get_fully_transformed_path().vertices, [(0, 0), (r2, r2), (0, 2 * r2), (-r2, r2)], atol=1e-15)
    path.points = [(0, 0)] * 4
    assert_allclose(trans_path.get_fully_transformed_path().vertices, [(0, 0), (r2, r2), (0, 2 * r2), (-r2, r2)], atol=1e-15)