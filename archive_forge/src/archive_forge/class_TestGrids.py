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
class TestGrids(TestPointsets):

    @pytest.mark.parametrize('shape', [(5, 5, 5), (5, 5, 5, 5), (5, 5, 5, 5, 5)])
    def test_from_image(self, shape):
        affine = np.diag([2, 3, 4, 1])
        img = SpatialImage(strided_scalar(shape), affine)
        grid = ps.Grid.from_image(img)
        grid_coords = grid.get_coords()
        assert grid.n_coords == prod(shape[:3])
        assert grid.dim == 3
        assert np.allclose(grid.affine, affine)
        assert np.allclose(grid_coords[0], [0, 0, 0])
        assert np.allclose(grid_coords[-1], [8, 12, 16])

    def test_from_mask(self):
        affine = np.diag([2, 3, 4, 1])
        mask = np.zeros((3, 3, 3))
        mask[1, 1, 1] = 1
        img = SpatialImage(mask, affine)
        grid = ps.Grid.from_mask(img)
        grid_coords = grid.get_coords()
        assert grid.n_coords == 1
        assert grid.dim == 3
        assert np.array_equal(grid_coords, [[2, 3, 4]])

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