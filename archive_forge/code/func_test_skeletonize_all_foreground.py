import numpy as np
import pytest
import scipy.ndimage as ndi
from skimage import io, draw
from skimage.data import binary_blobs
from skimage.morphology import skeletonize, skeletonize_3d
from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_, parametrize, fetch
def test_skeletonize_all_foreground():
    im = np.ones((3, 4), dtype=bool)
    assert_equal(skeletonize(im, method='lee'), np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=bool))