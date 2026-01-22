from itertools import product
import numpy as np
import param
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle, Wedge
from ...core.data import GridInterface
from ...core.spaces import HoloMap
from ...core.util import dimension_sanitizer, is_nan
from ..mixins import HeatMapMixin
from .element import ColorbarPlot
from .raster import QuadMeshPlot
from .util import filter_styles
@staticmethod
def _compute_separations(inner, outer, angles):
    """Compute x and y positions for separation lines for given angles.

        """
    return [np.array([[a, inner], [a, outer]]) for a in angles]