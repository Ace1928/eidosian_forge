import logging
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, collections, cm, colors, contour, ticker
import matplotlib.artist as martist
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.spines as mspines
import matplotlib.transforms as mtransforms
from matplotlib import _docstring
def _get_ticklocation_from_orientation(orientation):
    return _api.check_getitem({None: 'right', 'vertical': 'right', 'horizontal': 'bottom'}, orientation=orientation)