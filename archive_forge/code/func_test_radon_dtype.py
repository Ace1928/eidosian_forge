import itertools
import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import run_in_parallel
from skimage._shared.utils import _supported_float_type, convert_to_float
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, iradon_sart, rescale
def test_radon_dtype():
    img = convert_to_float(PHANTOM, False)
    img32 = img.astype(np.float32)
    assert radon(img).dtype == img.dtype
    assert radon(img32).dtype == img32.dtype