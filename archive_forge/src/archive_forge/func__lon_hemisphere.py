import itertools
import operator
import warnings
import matplotlib
import matplotlib.artist
import matplotlib.collections as mcollections
import matplotlib.text
import matplotlib.ticker as mticker
import matplotlib.transforms as mtrans
import numpy as np
import shapely.geometry as sgeom
import cartopy
from cartopy.crs import PlateCarree, Projection, _RectangularProjection
from cartopy.mpl.ticker import (
def _lon_hemisphere(longitude):
    """Return the hemisphere (E, W or '' for 0) for the given longitude."""
    lon_wrapped = (longitude + 180) % 360 - 180
    longitude = 180 if longitude > 0 and lon_wrapped == -180 else lon_wrapped
    if longitude > 0:
        hemisphere = 'E'
    elif longitude < 0:
        hemisphere = 'W'
    else:
        hemisphere = ''
    return hemisphere