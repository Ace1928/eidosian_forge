import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_transform_points():
    """
    Mostly tests the workaround for a specific problem.
    Problem report in: https://github.com/SciTools/cartopy/issues/232
    Fix covered in: https://github.com/SciTools/cartopy/pull/277
    """
    result = _CRS_ROB.transform_points(_CRS_PC, np.array([35.0]), np.array([70.0]))
    assert_array_almost_equal(result, [[2376187.2182271, 7275318.116298, 0]])
    result = _CRS_ROB.transform_points(_CRS_PC, np.array([35.0]), np.array([70.0]), np.array([0.0]))
    assert_array_almost_equal(result, [[2376187.2182271, 7275318.116298, 0]])
    result = _CRS_ROB.transform_points(_CRS_PC, np.array([np.nan]), np.array([70.0]))
    assert np.all(np.isnan(result))
    result = _CRS_ROB.transform_points(_CRS_PC, np.array([35.0]), np.array([np.nan]))
    assert np.all(np.isnan(result))
    x = np.array([10.0, 21.0, 0.0, 77.7, np.nan, 0.0])
    y = np.array([10.0, np.nan, 10.0, 77.7, 55.5, 0.0])
    z = np.array([10.0, 0.0, 0.0, np.nan, 55.5, 0.0])
    expect_result = np.array([[940422.591, 1069520.91, 10.0], [11.1, 11.2, 11.3], [0.0, 1069520.91213902, 0.0], [22.1, 22.2, 22.3], [33.1, 33.2, 33.3], [0.0, 0.0, 0.0]])
    result = _CRS_ROB.transform_points(_CRS_PC, x, y, z)
    assert result.shape == (6, 3)
    assert np.all(np.isnan(result[[1, 3, 4], :]))
    result[[1, 3, 4], :] = expect_result[[1, 3, 4], :]
    assert not np.any(np.isnan(result))
    assert np.allclose(result, expect_result)