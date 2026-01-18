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
def test_transformwrapper():
    t = mtransforms.TransformWrapper(mtransforms.Affine2D())
    with pytest.raises(ValueError, match='The input and output dims of the new child \\(1, 1\\) do not match those of current child \\(2, 2\\)'):
        t.set(scale.LogTransform(10))