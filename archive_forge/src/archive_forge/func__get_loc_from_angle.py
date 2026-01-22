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
def _get_loc_from_angle(self, angle):
    angle %= 360
    if angle > 180:
        angle -= 360
    if abs(angle) <= 45:
        loc = 'right'
    elif abs(angle) >= 135:
        loc = 'left'
    elif angle > 45:
        loc = 'top'
    else:
        loc = 'bottom'
    return loc