import matplotlib.pyplot as plt
import pytest
from cartopy.tests.conftest import _HAS_PYKDTREE_OR_SCIPY
import cartopy.crs as ccrs
from cartopy.io.ogc_clients import _OWSLIB_AVAILABLE
@pytest.mark.filterwarnings('ignore:TileMatrixLimits')
@pytest.mark.network
@pytest.mark.skipif(not _OWSLIB_AVAILABLE, reason='OWSLib is unavailable.')
@pytest.mark.mpl_image_compare(filename='wmts.png', tolerance=0.03)
@pytest.mark.xfail(reason='NASA servers are returning bad content metadata')
def test_wmts():
    ax = plt.axes(projection=ccrs.PlateCarree())
    url = 'https://map1c.vis.earthdata.nasa.gov/wmts-geo/wmts.cgi'
    ax.add_wmts(url, 'MODIS_Water_Mask')
    return ax.figure