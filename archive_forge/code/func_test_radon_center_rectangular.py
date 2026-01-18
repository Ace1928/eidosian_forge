import itertools
import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import run_in_parallel
from skimage._shared.utils import _supported_float_type, convert_to_float
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, iradon_sart, rescale
@pytest.mark.parametrize('shape', [(32, 16), (33, 17)])
@pytest.mark.parametrize('circle', [False])
@pytest.mark.parametrize('dtype', [np.float64, np.float32, np.uint8, bool])
@pytest.mark.parametrize('preserve_range', [False, True])
def test_radon_center_rectangular(shape, circle, dtype, preserve_range):
    check_radon_center(shape, circle, dtype, preserve_range)