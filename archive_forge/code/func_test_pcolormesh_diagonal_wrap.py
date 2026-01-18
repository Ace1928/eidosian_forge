import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_38
from cartopy.tests.conftest import requires_scipy
def test_pcolormesh_diagonal_wrap():
    xs = [[160, 170], [190, 200]]
    ys = [[-10, -10], [10, 10]]
    zs = [[0]]
    ax = plt.axes(projection=ccrs.PlateCarree())
    mesh = ax.pcolormesh(xs, ys, zs)
    assert hasattr(mesh, '_wrapped_collection_fix')