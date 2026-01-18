import pathlib
from tempfile import NamedTemporaryFile
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from skimage._shared.testing import fetch
from skimage.io import imread, imsave, reset_plugins, use_plugin
def test_tifffile_kwarg_passthrough():
    img = imread(fetch('data/multipage.tif'), key=[1], is_ome=True)
    assert img.shape == (15, 10), img.shape