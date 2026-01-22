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
class AssertingNonAffineTransform(mtransforms.Transform):
    """
        This transform raises an assertion error when called when it
        shouldn't be and ``self.raise_on_transform`` is True.

        """
    input_dims = output_dims = 2
    is_affine = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raise_on_transform = False
        self.underlying_transform = mtransforms.Affine2D().scale(10, 10)

    def transform_path_non_affine(self, path):
        assert not self.raise_on_transform, 'Invalidated affine part of transform unnecessarily.'
        return self.underlying_transform.transform_path(path)
    transform_path = transform_path_non_affine

    def transform_non_affine(self, path):
        assert not self.raise_on_transform, 'Invalidated affine part of transform unnecessarily.'
        return self.underlying_transform.transform(path)
    transform = transform_non_affine