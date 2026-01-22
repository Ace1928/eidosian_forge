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
class InvertedHammerTransform(_GeoTransform):

    @_api.rename_parameter('3.8', 'xy', 'values')
    def transform_non_affine(self, values):
        x, y = values.T
        z = np.sqrt(1 - (x / 4) ** 2 - (y / 2) ** 2)
        longitude = 2 * np.arctan(z * x / (2 * (2 * z ** 2 - 1)))
        latitude = np.arcsin(y * z)
        return np.column_stack([longitude, latitude])

    def inverted(self):
        return HammerAxes.HammerTransform(self._resolution)