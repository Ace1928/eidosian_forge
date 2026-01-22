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
class Mollweide(_WarpedRectangularProjection):
    """
    A Mollweide projection.

    This projection is pseudocylindrical, and equal area. Parallels are
    unequally-spaced straight lines, while meridians are elliptical arcs up to
    semicircles on the edges. Poles are points.

    It is commonly used for world maps, or interrupted with several central
    meridians.

    """
    _handles_ellipses = False

    def __init__(self, central_longitude=0, globe=None, false_easting=None, false_northing=None):
        """
        Parameters
        ----------
        central_longitude: float, optional
            The central longitude. Defaults to 0.
        false_easting: float, optional
            X offset from planar origin in metres. Defaults to 0.
        false_northing: float, optional
            Y offset from planar origin in metres. Defaults to 0.
        globe: :class:`cartopy.crs.Globe`, optional
            If omitted, a default globe is created.

            .. note::
                This projection does not handle elliptical globes.

        """
        proj4_params = [('proj', 'moll'), ('lon_0', central_longitude)]
        super().__init__(proj4_params, central_longitude, false_easting=false_easting, false_northing=false_northing, globe=globe)
        self.threshold = 100000.0