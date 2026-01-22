from collections import defaultdict
import functools
import itertools
import math
import textwrap
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, _preprocess_data
import matplotlib.artist as martist
import matplotlib.axes as maxes
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.container as mcontainer
import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes
from matplotlib.axes._base import _axis_method_wrapper, _process_plot_format
from matplotlib.transforms import Bbox
from matplotlib.tri._triangulation import Triangulation
from . import art3d
from . import proj3d
from . import axis3d
def _calc_coord(self, xv, yv, renderer=None):
    """
        Given the 2D view coordinates, find the point on the nearest axis pane
        that lies directly below those coordinates. Returns a 3D point in data
        coordinates.
        """
    if self._focal_length == np.inf:
        zv = 1
    else:
        zv = -1 / self._focal_length
    p1 = np.array(proj3d.inv_transform(xv, yv, zv, self.invM)).ravel()
    vec = self._get_camera_loc() - p1
    pane_locs = []
    for axis in self._axis_map.values():
        xys, loc = axis.active_pane(renderer)
        pane_locs.append(loc)
    scales = np.zeros(3)
    for i in range(3):
        if vec[i] == 0:
            scales[i] = np.inf
        else:
            scales[i] = (p1[i] - pane_locs[i]) / vec[i]
    pane_idx = np.argmin(abs(scales))
    scale = scales[pane_idx]
    p2 = p1 - scale * vec
    return (p2, pane_idx)