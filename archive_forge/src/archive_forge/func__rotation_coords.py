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
def _rotation_coords(self):
    """
        Return the rotation angles as a string.
        """
    norm_elev = art3d._norm_angle(self.elev)
    norm_azim = art3d._norm_angle(self.azim)
    norm_roll = art3d._norm_angle(self.roll)
    coords = f'elevation={norm_elev:.0f}°, azimuth={norm_azim:.0f}°, roll={norm_roll:.0f}°'.replace('-', '−')
    return coords