import inspect
import numpy as np
import pytest
from skimage import data, morphology, util
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.filters import rank
from skimage.filters.rank import __all__ as all_rank_filters
from skimage.filters.rank import __3Dfilters as _3d_rank_filters
from skimage.filters.rank import subtract_mean
from skimage.morphology import ball, disk, gray
from skimage.util import img_as_float, img_as_ubyte
def test_otsu_edge_case():
    footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    img = np.array([[0, 41, 0], [30, 81, 106], [0, 147, 0]], dtype=np.uint8)
    result = rank.otsu(img, footprint)
    assert result[1, 1] in [41, 81]
    img = np.array([[0, 214, 0], [229, 104, 141], [0, 172, 0]], dtype=np.uint8)
    result = rank.otsu(img, footprint)
    assert result[1, 1] in [141, 172]