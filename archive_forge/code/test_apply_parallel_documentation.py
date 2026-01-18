import numpy as np
from skimage._shared.testing import assert_array_almost_equal, assert_equal
from skimage import color, data, img_as_float
from skimage.filters import threshold_local, gaussian
from skimage.util.apply_parallel import apply_parallel
import pytest
Test channel_axis combinations.

    For depth and chunks, test in three ways:
    1.) scalar (to be applied over all axes)
    2.) tuple of length ``image.ndim - 1`` corresponding to spatial axes
    3.) tuple of length ``image.ndim`` corresponding to all axes
    