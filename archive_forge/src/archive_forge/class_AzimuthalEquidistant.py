from abc import ABCMeta
from collections import OrderedDict
from functools import lru_cache
import io
import json
import math
import warnings
import numpy as np
from pyproj import Transformer
from pyproj.exceptions import ProjError
import shapely.geometry as sgeom
from shapely.prepared import prep
import cartopy.trace
class AzimuthalEquidistant(Projection):
    """
    An Azimuthal Equidistant projection

    This projection provides accurate angles about and distances through the
    central position. Other angles, distances, or areas may be distorted.
    """
    _wrappable = True

    def __init__(self, central_longitude=0.0, central_latitude=0.0, false_easting=0.0, false_northing=0.0, globe=None):
        """
        Parameters
        ----------
        central_longitude: optional
            The true longitude of the central meridian in degrees.
            Defaults to 0.
        central_latitude: optional
            The true latitude of the planar origin in degrees.
            Defaults to 0.
        false_easting: optional
            X offset from the planar origin in metres. Defaults to 0.
        false_northing: optional
            Y offset from the planar origin in metres. Defaults to 0.
        globe: optional
            An instance of :class:`cartopy.crs.Globe`. If omitted, a default
            globe is created.

        """
        proj4_params = [('proj', 'aeqd'), ('lon_0', central_longitude), ('lat_0', central_latitude), ('x_0', false_easting), ('y_0', false_northing)]
        super().__init__(proj4_params, globe=globe)
        a = float(self.ellipsoid.semi_major_metre or WGS84_SEMIMAJOR_AXIS)
        b = float(self.ellipsoid.semi_minor_metre or a)
        coords = _ellipse_boundary(a * np.pi, b * np.pi, false_easting, false_northing, 61)
        self._boundary = sgeom.LinearRing(coords.T)
        mins = np.min(coords, axis=1)
        maxs = np.max(coords, axis=1)
        self._x_limits = (mins[0], maxs[0])
        self._y_limits = (mins[1], maxs[1])
        self.threshold = 100000.0

    @property
    def boundary(self):
        return self._boundary

    @property
    def x_limits(self):
        return self._x_limits

    @property
    def y_limits(self):
        return self._y_limits