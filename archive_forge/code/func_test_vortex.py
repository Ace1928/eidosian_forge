from packaging.version import Version
import numpy as np
import skimage.data as data
from skimage.data._fetchers import _image_fetcher
from skimage import io
from skimage._shared.testing import assert_equal, assert_almost_equal, fetch
import os
import pytest
def test_vortex():
    fetch('data/pivchallenge-B-B001_1.tif')
    fetch('data/pivchallenge-B-B001_2.tif')
    image0, image1 = data.vortex()
    for image in [image0, image1]:
        assert image.shape == (512, 512)