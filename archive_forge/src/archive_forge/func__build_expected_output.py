import numpy as np
import pytest
from scipy import ndimage as ndi
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from skimage import color, data, transform
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import fetch, assert_stacklevel
from skimage.morphology import gray, footprints
from skimage.util import img_as_uint, img_as_ubyte
def _build_expected_output(self):
    footprints_2D = (footprints.square, footprints.diamond, footprints.disk, footprints.star)
    image = img_as_ubyte(transform.downscale_local_mean(color.rgb2gray(data.coffee()), (20, 20)))
    output = {}
    for n in range(1, 4):
        for strel in footprints_2D:
            for func in gray_morphology_funcs:
                key = f'{strel.__name__}_{n}_{func.__name__}'
                output[key] = func(image, strel(n))
    return output