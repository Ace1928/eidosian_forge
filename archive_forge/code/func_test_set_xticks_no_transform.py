import matplotlib.pyplot as plt
import pytest
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='xticks_no_transform.png')
def test_set_xticks_no_transform():
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines('110m')
    ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol=''))
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticks([-135, -45, 45, 135], minor=True)
    return ax.figure