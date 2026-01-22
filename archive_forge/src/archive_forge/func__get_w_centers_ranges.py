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
def _get_w_centers_ranges(self):
    """Get 3D world centers and axis ranges."""
    minx, maxx, miny, maxy, minz, maxz = self.get_w_lims()
    cx = (maxx + minx) / 2
    cy = (maxy + miny) / 2
    cz = (maxz + minz) / 2
    dx = maxx - minx
    dy = maxy - miny
    dz = maxz - minz
    return (cx, cy, cz, dx, dy, dz)