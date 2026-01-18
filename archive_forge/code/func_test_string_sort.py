import os
import itertools
import numpy as np
import imageio.v3 as iio3
from skimage import data_dir
from skimage.io.collection import ImageCollection, MultiImage, alphanumeric_key
from skimage.io import reset_plugins
from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_allclose, fetch
import pytest
def test_string_sort():
    filenames = ['f9.10.png', 'f9.9.png', 'f10.10.png', 'f10.9.png', 'e9.png', 'e10.png', 'em.png']
    expected_filenames = ['e9.png', 'e10.png', 'em.png', 'f9.9.png', 'f9.10.png', 'f10.9.png', 'f10.10.png']
    sorted_filenames = sorted(filenames, key=alphanumeric_key)
    assert_equal(expected_filenames, sorted_filenames)