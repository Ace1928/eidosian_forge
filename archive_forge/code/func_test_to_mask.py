from math import prod
from pathlib import Path
from unittest import skipUnless
import numpy as np
import pytest
from nibabel import pointset as ps
from nibabel.affines import apply_affine
from nibabel.arrayproxy import ArrayProxy
from nibabel.fileslice import strided_scalar
from nibabel.onetime import auto_attr
from nibabel.optpkg import optional_package
from nibabel.spatialimages import SpatialImage
from nibabel.tests.nibabel_data import get_nibabel_data
def test_to_mask(self):
    coords = np.array([[1, 1, 1]])
    grid = ps.Grid(coords)
    mask_img = grid.to_mask()
    assert mask_img.shape == (2, 2, 2)
    assert np.array_equal(mask_img.get_fdata(), [[[0, 0], [0, 0]], [[0, 0], [0, 1]]])
    assert np.array_equal(mask_img.affine, np.eye(4))
    mask_img = grid.to_mask(shape=(3, 3, 3))
    assert mask_img.shape == (3, 3, 3)
    assert np.array_equal(mask_img.get_fdata(), [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    assert np.array_equal(mask_img.affine, np.eye(4))