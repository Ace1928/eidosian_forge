import matplotlib as mpl
from matplotlib.ticker import Formatter, MaxNLocator
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
@classmethod
def _fix_lons(cls, lons):
    if isinstance(lons, list):
        return [cls._fix_lons(lon) for lon in lons]
    p180 = lons == 180
    m180 = lons == -180
    lons = (lons + 180) % 360 - 180
    for mp180, value in [(m180, -180), (p180, 180)]:
        if np.any(mp180):
            if isinstance(lons, np.ndarray):
                lons = np.where(mp180, value, lons)
            else:
                lons = value
    return lons