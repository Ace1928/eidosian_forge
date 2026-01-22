import matplotlib as mpl
from matplotlib.ticker import Formatter, MaxNLocator
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
def _guess_steps(self, vmin, vmax):
    vmin = max(vmin, -90.0)
    vmax = min(vmax, 90.0)
    LongitudeLocator._guess_steps(self, vmin, vmax)