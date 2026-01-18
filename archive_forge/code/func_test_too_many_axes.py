import numpy as np
import pytest
from skimage.util import slice_along_axes
def test_too_many_axes():
    data = np.empty((10, 10))
    with pytest.raises(ValueError):
        slice_along_axes(data, [(0, 1), (0, 1), (0, 1)])