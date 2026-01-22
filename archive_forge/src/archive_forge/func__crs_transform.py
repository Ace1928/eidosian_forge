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
def _crs_transform(self):
    """
        Get the drawing transform for our gridlines.

        Note
        ----
            The drawing transform depends on the transform of our 'axes', so
            it may change dynamically.

        """
    transform = self.crs
    if not isinstance(transform, mtrans.Transform):
        transform = transform._as_mpl_transform(self.axes)
    return transform