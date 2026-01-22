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
def _get_loc_from_spine_intersection(self, spines_specs, xylabel, x, y):
    """Get the loc the intersection of a gridline with a spine

        Defaults to "geo".
        """
    if xylabel == 'x':
        sides = ['bottom', 'top', 'left', 'right']
    else:
        sides = ['left', 'right', 'bottom', 'top']
    for side in sides:
        xy = x if side in ['left', 'right'] else y
        coords = np.round(spines_specs[side]['coords'], 2)
        if round(xy, 2) in coords:
            return side
    return 'geo'