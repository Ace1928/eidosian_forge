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
class AitoffTransform(_GeoTransform):
    """The base Aitoff transform."""

    @_api.rename_parameter('3.8', 'll', 'values')
    def transform_non_affine(self, values):
        longitude, latitude = values.T
        half_long = longitude / 2.0
        cos_latitude = np.cos(latitude)
        alpha = np.arccos(cos_latitude * np.cos(half_long))
        sinc_alpha = np.sinc(alpha / np.pi)
        x = cos_latitude * np.sin(half_long) / sinc_alpha
        y = np.sin(latitude) / sinc_alpha
        return np.column_stack([x, y])

    def inverted(self):
        return AitoffAxes.InvertedAitoffTransform(self._resolution)