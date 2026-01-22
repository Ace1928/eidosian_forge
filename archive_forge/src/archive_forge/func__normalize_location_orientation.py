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
def _normalize_location_orientation(location, orientation):
    if location is None:
        location = _get_ticklocation_from_orientation(orientation)
    loc_settings = _api.check_getitem({'left': {'location': 'left', 'anchor': (1.0, 0.5), 'panchor': (0.0, 0.5), 'pad': 0.1}, 'right': {'location': 'right', 'anchor': (0.0, 0.5), 'panchor': (1.0, 0.5), 'pad': 0.05}, 'top': {'location': 'top', 'anchor': (0.5, 0.0), 'panchor': (0.5, 1.0), 'pad': 0.05}, 'bottom': {'location': 'bottom', 'anchor': (0.5, 1.0), 'panchor': (0.5, 0.0), 'pad': 0.15}}, location=location)
    loc_settings['orientation'] = _get_orientation_from_location(location)
    if orientation is not None and orientation != loc_settings['orientation']:
        raise TypeError('location and orientation are mutually exclusive')
    return loc_settings