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
def test_transformed_patch_path():
    trans = mtransforms.Affine2D()
    patch = mpatches.Wedge((0, 0), 1, 45, 135, transform=trans)
    tpatch = mtransforms.TransformedPatchPath(patch)
    points = tpatch.get_fully_transformed_path().vertices
    trans.scale(2)
    assert_allclose(tpatch.get_fully_transformed_path().vertices, points * 2)
    patch.set_radius(0.5)
    assert_allclose(tpatch.get_fully_transformed_path().vertices, points)