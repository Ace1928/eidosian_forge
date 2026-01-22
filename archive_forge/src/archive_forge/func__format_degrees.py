import matplotlib as mpl
from matplotlib.ticker import Formatter, MaxNLocator
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
def _format_degrees(self, deg):
    return _PlateCarreeFormatter._format_degrees(self, self._fix_lons(deg))