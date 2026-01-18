import pytest
import cartopy.feature as cfeature
def test_scale_from_extent(self):
    small_scale = auto_land.scaler.scale_from_extent(small_extent)
    medium_scale = auto_land.scaler.scale_from_extent(medium_extent)
    large_scale = auto_land.scaler.scale_from_extent(large_extent)
    assert small_scale == '10m'
    assert medium_scale == '50m'
    assert large_scale == '110m'