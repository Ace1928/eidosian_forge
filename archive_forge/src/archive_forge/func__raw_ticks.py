import matplotlib as mpl
from matplotlib.ticker import Formatter, MaxNLocator
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
def _raw_ticks(self, vmin, vmax):
    ticks = LongitudeLocator._raw_ticks(self, vmin, vmax)
    return [t for t in ticks if -90 <= t <= 90]