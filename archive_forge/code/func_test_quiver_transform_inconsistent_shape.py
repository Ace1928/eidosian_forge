from unittest import mock
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
def test_quiver_transform_inconsistent_shape(self):
    with pytest.raises(ValueError):
        self.ax.quiver(self.x, self.y, self.u.ravel(), self.v.ravel(), transform=self.rp)