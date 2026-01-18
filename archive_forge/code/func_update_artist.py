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
def update_artist(artist, renderer):
    artist.update_bbox_position_size(renderer)
    this_patch = artist.get_bbox_patch()
    this_path = this_patch.get_path().transformed(this_patch.get_transform())
    return this_path