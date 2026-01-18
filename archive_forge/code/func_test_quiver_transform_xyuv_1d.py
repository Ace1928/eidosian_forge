from unittest import mock
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
def test_quiver_transform_xyuv_1d(self):
    with mock.patch('matplotlib.axes.Axes.quiver') as patch:
        self.ax.quiver(self.x2d.ravel(), self.y2d.ravel(), self.u.ravel(), self.v.ravel(), transform=self.rp)
    args, kwargs = patch.call_args
    assert len(args) == 4
    assert sorted(kwargs.keys()) == ['transform']
    shapes = [arg.shape for arg in args]
    assert shapes == [(70,)] * 4