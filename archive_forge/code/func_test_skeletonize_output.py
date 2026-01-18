import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.ndimage import correlate
from skimage import draw
from skimage._shared.testing import fetch
from skimage.io import imread
from skimage.morphology import medial_axis, skeletonize, thin
from skimage.morphology._skeletonize import G123_LUT, G123P_LUT, _generate_thin_luts
def test_skeletonize_output(self):
    im = imread(fetch('data/bw_text.png'), as_gray=True)
    im = im == 0
    result = skeletonize(im)
    expected = np.load(fetch('data/bw_text_skeleton.npy'))
    assert_array_equal(result, expected)