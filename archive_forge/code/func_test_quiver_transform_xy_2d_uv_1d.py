from unittest import mock
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
def test_quiver_transform_xy_2d_uv_1d(self):
    with pytest.raises(ValueError):
        self.ax.quiver(self.x2d, self.y2d, self.u.ravel(), self.v.ravel(), transform=self.rp)