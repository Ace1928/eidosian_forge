import matplotlib.pyplot as plt
import pytest
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='yticks_cylindrical.png')
def test_set_yticks_cylindrical():
    ax = plt.axes(projection=ccrs.Mercator(min_latitude=-85, max_latitude=85))
    ax.coastlines('110m')
    ax.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol=''))
    ax.set_yticks([-60, -30, 0, 30, 60], crs=ccrs.PlateCarree())
    ax.set_yticks([-75, -45, -15, 15, 45, 75], minor=True, crs=ccrs.PlateCarree())
    return ax.figure