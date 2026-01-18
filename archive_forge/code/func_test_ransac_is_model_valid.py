import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (
from skimage.measure import CircleModel, EllipseModel, LineModelND, ransac
from skimage.measure.fit import _dynamic_max_trials
from skimage.transform import AffineTransform
def test_ransac_is_model_valid():

    def is_model_valid(model, data):
        return False
    with expected_warnings(['No inliers found']):
        model, inliers = ransac(np.empty((10, 2)), LineModelND, 2, np.inf, is_model_valid=is_model_valid, rng=1)
    assert_equal(model, None)
    assert_equal(inliers, None)