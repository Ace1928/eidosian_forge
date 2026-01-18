import numpy as np
import pytest
import scipy.ndimage as ndi
from skimage import io, draw
from skimage.data import binary_blobs
from skimage.morphology import skeletonize, skeletonize_3d
from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_, parametrize, fetch
def test_skeletonize_wrong_dim():
    im = np.zeros(5, dtype=bool)
    with testing.raises(ValueError):
        skeletonize(im, method='lee')
    im = np.zeros((5, 5, 5, 5), dtype=bool)
    with testing.raises(ValueError):
        skeletonize(im, method='lee')