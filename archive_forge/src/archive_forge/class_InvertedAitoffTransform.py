import numpy as np
import matplotlib as mpl
from matplotlib import _api
from matplotlib.axes import Axes
import matplotlib.axis as maxis
from matplotlib.patches import Circle
from matplotlib.path import Path
import matplotlib.spines as mspines
from matplotlib.ticker import (
from matplotlib.transforms import Affine2D, BboxTransformTo, Transform
class InvertedAitoffTransform(_GeoTransform):

    @_api.rename_parameter('3.8', 'xy', 'values')
    def transform_non_affine(self, values):
        return np.full_like(values, np.nan)

    def inverted(self):
        return AitoffAxes.AitoffTransform(self._resolution)