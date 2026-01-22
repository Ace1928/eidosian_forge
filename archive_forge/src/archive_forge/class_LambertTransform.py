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
class LambertTransform(_GeoTransform):
    """The base Lambert transform."""

    def __init__(self, center_longitude, center_latitude, resolution):
        """
            Create a new Lambert transform.  Resolution is the number of steps
            to interpolate between each input line segment to approximate its
            path in curved Lambert space.
            """
        _GeoTransform.__init__(self, resolution)
        self._center_longitude = center_longitude
        self._center_latitude = center_latitude

    @_api.rename_parameter('3.8', 'll', 'values')
    def transform_non_affine(self, values):
        longitude, latitude = values.T
        clong = self._center_longitude
        clat = self._center_latitude
        cos_lat = np.cos(latitude)
        sin_lat = np.sin(latitude)
        diff_long = longitude - clong
        cos_diff_long = np.cos(diff_long)
        inner_k = np.maximum(1 + np.sin(clat) * sin_lat + np.cos(clat) * cos_lat * cos_diff_long, 1e-15)
        k = np.sqrt(2 / inner_k)
        x = k * cos_lat * np.sin(diff_long)
        y = k * (np.cos(clat) * sin_lat - np.sin(clat) * cos_lat * cos_diff_long)
        return np.column_stack([x, y])

    def inverted(self):
        return LambertAxes.InvertedLambertTransform(self._center_longitude, self._center_latitude, self._resolution)