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
class NearsidePerspective(_Satellite):
    """
    Perspective view looking directly down from above a point on the globe.

    In this projection, the projected coordinates are x and y measured from
    the origin of a plane tangent to the Earth directly below the perspective
    point (e.g. a satellite).

    """
    _handles_ellipses = False

    def __init__(self, central_longitude=0.0, central_latitude=0.0, satellite_height=35785831, false_easting=0, false_northing=0, globe=None):
        """
        Parameters
        ----------
        central_longitude: float, optional
            The central longitude. Defaults to 0.
        central_latitude: float, optional
            The central latitude. Defaults to 0.
        satellite_height: float, optional
            The height of the satellite. Defaults to 35785831 meters
            (true geostationary orbit).
        false_easting:
            X offset from planar origin in metres. Defaults to 0.
        false_northing:
            Y offset from planar origin in metres. Defaults to 0.
        globe: :class:`cartopy.crs.Globe`, optional
            If omitted, a default globe is created.

            .. note::
                This projection does not handle elliptical globes.

        """
        super().__init__(projection='nsper', satellite_height=satellite_height, central_longitude=central_longitude, central_latitude=central_latitude, false_easting=false_easting, false_northing=false_northing, globe=globe)
        a = self.ellipsoid.semi_major_metre or WGS84_SEMIMAJOR_AXIS
        h = float(satellite_height)
        max_x = a * np.sqrt(h / (2 * a + h))
        coords = _ellipse_boundary(max_x, max_x, false_easting, false_northing, 61)
        self._set_boundary(coords)