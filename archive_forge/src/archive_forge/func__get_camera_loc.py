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
def _get_camera_loc(self):
    """
        Returns the current camera location in data coordinates.
        """
    cx, cy, cz, dx, dy, dz = self._get_w_centers_ranges()
    c = np.array([cx, cy, cz])
    r = np.array([dx, dy, dz])
    if self._focal_length == np.inf:
        focal_length = 1000000000.0
    else:
        focal_length = self._focal_length
    eye = c + self._view_w * self._dist * r / self._box_aspect * focal_length
    return eye