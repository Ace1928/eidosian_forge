import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.mpl import _MPL_38
from cartopy.tests.conftest import requires_scipy
@pytest.mark.filterwarnings('ignore:Unable to determine extent')
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='simple_global.png')
def test_simple_global():
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    return ax.figure