from __future__ import annotations
import typing
import numpy as np
import pandas as pd
from .._utils import SIZE_FACTOR, to_rgba
from ..doctools import document
from ..exceptions import PlotnineError
from .geom import geom
from .geom_point import geom_point
from .geom_polygon import geom_polygon
def PolygonPatch(obj: Polygon) -> PathPatch:
    """
    Return a Matplotlib patch from a Polygon/MultiPolygon Geometry

    Parameters
    ----------
    obj : shapley.geometry.Polygon | shapley.geometry.MultiPolygon
        A Polygon or MultiPolygon to create a patch for description

    Returns
    -------
    result : matplotlib.patches.PathPatch
        A patch representing the shapely geometry

    Notes
    -----
    This functionality was originally provided by the descartes package
    by Sean Gillies (BSD license, https://pypi.org/project/descartes)
    which is nolonger being maintained.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    def cw_coords(ring: LinearRing) -> npt.NDArray[Any]:
        """
        Return Clockwise array coordinates

        Parameters
        ----------
        ring: shapely.geometry.polygon.LinearRing
            LinearRing

        Returns
        -------
        out: ndarray
            (n x 2) array of coordinate points.
        """
        if ring.is_ccw:
            return np.asarray(ring.coords)[:, :2][::-1]
        return np.asarray(ring.coords)[:, :2]

    def ccw_coords(ring: LinearRing) -> npt.NDArray[Any]:
        """
        Return Counter Clockwise array coordinates

        Parameters
        ----------
        ring: shapely.geometry.polygon.LinearRing
            LinearRing

        Returns
        -------
        out: ndarray
            (n x 2) array of coordinate points.
        """
        if ring.is_ccw:
            return np.asarray(ring.coords)[:, :2]
        return np.asarray(ring.coords)[:, :2][::-1]
    if obj.geom_type == 'Polygon':
        _exterior = [Path(cw_coords(obj.exterior))]
        _interior = [Path(ccw_coords(ring)) for ring in obj.interiors]
    else:
        _exterior = []
        _interior = []
        for p in obj.geoms:
            _exterior.append(Path(cw_coords(p.exterior)))
            _interior.extend([Path(ccw_coords(ring)) for ring in p.interiors])
    path = Path.make_compound_path(*_exterior, *_interior)
    return PathPatch(path)