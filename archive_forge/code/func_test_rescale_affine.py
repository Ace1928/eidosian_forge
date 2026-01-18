from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from ..affines import (
from ..eulerangles import euler2mat
from ..orientations import aff2axcodes
def test_rescale_affine():
    rng = np.random.RandomState(20200415)
    orig_shape = rng.randint(low=20, high=512, size=(3,))
    orig_aff = np.eye(4)
    orig_aff[:3, :] = rng.normal(size=(3, 4))
    orig_zooms = voxel_sizes(orig_aff)
    orig_axcodes = aff2axcodes(orig_aff)
    orig_centroid = apply_affine(orig_aff, (orig_shape - 1) // 2)
    for new_shape in (None, tuple(orig_shape), (256, 256, 256), (64, 64, 40)):
        for new_zooms in ((1, 1, 1), (2, 2, 3), (0.5, 0.5, 0.5)):
            new_aff = rescale_affine(orig_aff, orig_shape, new_zooms, new_shape)
            assert aff2axcodes(new_aff) == orig_axcodes
            if new_shape is None:
                new_shape = tuple(orig_shape)
            new_centroid = apply_affine(new_aff, (np.array(new_shape) - 1) // 2)
            assert_almost_equal(new_centroid, orig_centroid)