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
class InvertedMollweideTransform(_GeoTransform):

    @_api.rename_parameter('3.8', 'xy', 'values')
    def transform_non_affine(self, values):
        x, y = values.T
        theta = np.arcsin(y / np.sqrt(2))
        longitude = np.pi / (2 * np.sqrt(2)) * x / np.cos(theta)
        latitude = np.arcsin((2 * theta + np.sin(2 * theta)) / np.pi)
        return np.column_stack([longitude, latitude])

    def inverted(self):
        return MollweideAxes.MollweideTransform(self._resolution)