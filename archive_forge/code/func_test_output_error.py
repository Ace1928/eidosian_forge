import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage._shared.utils import _supported_float_type
from skimage._shared.testing import assert_stacklevel
from skimage.filters import difference_of_gaussians, gaussian
def test_output_error():
    image = np.arange(9, dtype=np.float32).reshape((3, 3))
    out = np.zeros_like(image, dtype=np.uint8)
    with pytest.raises(ValueError, match='dtype of `out` must be float'):
        gaussian(image, sigma=1, out=out, preserve_range=True)