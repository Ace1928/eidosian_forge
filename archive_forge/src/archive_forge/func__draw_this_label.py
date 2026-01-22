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
def _draw_this_label(self, xylabel, loc):
    """Should I draw this kind of label here?"""
    draw_labels = getattr(self, loc + '_labels')
    if draw_labels is True and loc != 'geo':
        draw_labels = 'x' if loc in ['top', 'bottom'] else 'y'
    if not draw_labels:
        return False
    if isinstance(draw_labels, str):
        draw_labels = [draw_labels]
    if isinstance(draw_labels, list) and xylabel not in draw_labels:
        return False
    return True