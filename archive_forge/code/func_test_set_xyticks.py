import matplotlib.pyplot as plt
import pytest
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
@pytest.mark.natural_earth
@pytest.mark.mpl_image_compare(filename='xyticks.png')
def test_set_xyticks():
    fig = plt.figure(figsize=(10, 10))
    projections = (ccrs.PlateCarree(), ccrs.Mercator(), ccrs.TransverseMercator(approx=False))
    x = -3.275024
    y = 50.753998
    for i, prj in enumerate(projections, 1):
        ax = fig.add_subplot(3, 1, i, projection=prj)
        ax.set_extent([-12.5, 4, 49, 60], ccrs.Geodetic())
        ax.coastlines('110m')
        p, q = prj.transform_point(x, y, ccrs.Geodetic())
        ax.set_xticks([p])
        ax.set_yticks([q])
    return fig