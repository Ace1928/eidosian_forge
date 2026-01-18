from unittest import mock
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
def test_quiver_transform_xy_1d_uv_2d(self):
    with mock.patch('matplotlib.axes.Axes.quiver') as patch:
        self.ax.quiver(self.x, self.y, self.u, self.v, transform=self.rp)
    args, kwargs = patch.call_args
    assert len(args) == 4
    assert sorted(kwargs.keys()) == ['transform']
    shapes = [arg.shape for arg in args]
    assert shapes == [(7, 10)] * 4