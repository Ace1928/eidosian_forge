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
Test that shape of output image in deconvolution is same as input.

    This addresses issue #1172.
    