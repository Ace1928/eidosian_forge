from packaging.version import Version
import numpy as np
import skimage.data as data
from skimage.data._fetchers import _image_fetcher
from skimage import io
from skimage._shared.testing import assert_equal, assert_almost_equal, fetch
import os
import pytest
def test_skin():
    """Test that "skin" image can be loaded.

    Needs internet connection.
    """
    skin = data.skin()
    assert skin.ndim == 3