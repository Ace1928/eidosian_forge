import numpy as np
import pytest
from scipy import ndimage as ndi
from scipy.signal import convolve2d, convolve
from skimage import restoration, util
from skimage._shared import filters
from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.data import astronaut, camera
from skimage.restoration import uft
def test_unsupervised_wiener_deprecated_user_param():
    psf = np.ones((5, 5), dtype=float) / 25
    data = convolve2d(test_img, psf, 'same')
    otf = uft.ir2tf(psf, data.shape, is_real=False)
    _, laplacian = uft.laplacian(2, data.shape)
    restoration.unsupervised_wiener(data, otf, reg=laplacian, is_real=False, user_params={'max_num_iter': 300, 'min_num_iter': 30}, rng=5)