import itertools
import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import run_in_parallel
from skimage._shared.utils import _supported_float_type, convert_to_float
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, iradon_sart, rescale
def test_order_angles_golden_ratio():
    from skimage.transform.radon_transform import order_angles_golden_ratio
    np.random.seed(1231)
    lengths = [1, 4, 10, 180]
    for l in lengths:
        theta_ordered = np.linspace(0, 180, l, endpoint=False)
        theta_random = np.random.uniform(0, 180, l)
        for theta in (theta_random, theta_ordered):
            indices = [x for x in order_angles_golden_ratio(theta)]
            assert len(indices) == len(set(indices))