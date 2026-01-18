import numpy as np
import pytest
from skimage._shared.testing import expected_warnings, run_in_parallel
from skimage.feature import (
from skimage.transform import integral_image
def test_uniform_properties(self):
    im = np.ones((4, 4), dtype=np.uint8)
    result = graycomatrix(im, [1, 2, 8], [0, np.pi / 2], 4, normed=True, symmetric=True)
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        graycoprops(result, prop)