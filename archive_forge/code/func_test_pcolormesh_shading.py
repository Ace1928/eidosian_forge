import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_38
from cartopy.tests.conftest import requires_scipy
@pytest.mark.parametrize('shading, input_size, expected', [pytest.param('auto', 3, 4, id='auto same size'), pytest.param('auto', 4, 4, id='auto input larger'), pytest.param('nearest', 3, 4, id='nearest same size'), pytest.param('nearest', 4, 4, id='nearest input larger'), pytest.param('flat', 4, 4, id='flat input larger'), pytest.param('gouraud', 3, 3, id='gouraud same size')])
def test_pcolormesh_shading(shading, input_size, expected):
    ax = plt.axes(projection=ccrs.PlateCarree())
    x = np.arange(input_size)
    y = np.arange(input_size)
    d = np.zeros((3, 3))
    coll = ax.pcolormesh(x, y, d, shading=shading)
    assert coll._coordinates.shape == (expected, expected, 2)