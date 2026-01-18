import numpy as np
from shapely import lib
from shapely._enum import ParamEnum
from shapely.algorithms._oriented_envelope import _oriented_envelope_min_area_vectorized
from shapely.decorators import multithreading_enabled, requires_geos
@multithreading_enabled
def voronoi_polygons(geometry, tolerance=0.0, extend_to=None, only_edges=False, **kwargs):
    """Computes a Voronoi diagram from the vertices of an input geometry.

    The output is a geometrycollection containing polygons (default)
    or linestrings (see only_edges). Returns empty if an input geometry
    contains less than 2 vertices or if the provided extent has zero area.

    Parameters
    ----------
    geometry : Geometry or array_like
    tolerance : float or array_like, default 0.0
        Snap input vertices together if their distance is less than this value.
    extend_to : Geometry or array_like, optional
        If provided, the diagram will be extended to cover the envelope of this
        geometry (unless this envelope is smaller than the input geometry).
    only_edges : bool or array_like, default False
        If set to True, the triangulation will return a collection of
        linestrings instead of polygons.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> from shapely import LineString, MultiPoint, normalize, Point
    >>> points = MultiPoint([(2, 2), (4, 2)])
    >>> normalize(voronoi_polygons(points))
    <GEOMETRYCOLLECTION (POLYGON ((3 0, 3 4, 6 4, 6 0, 3 0)), POLYGON ((0 0, 0 4...>
    >>> voronoi_polygons(points, only_edges=True)
    <LINESTRING (3 4, 3 0)>
    >>> voronoi_polygons(MultiPoint([(2, 2), (4, 2), (4.2, 2)]), 0.5, only_edges=True)
    <LINESTRING (3 4.2, 3 -0.2)>
    >>> voronoi_polygons(points, extend_to=LineString([(0, 0), (10, 10)]), only_edges=True)
    <LINESTRING (3 10, 3 0)>
    >>> voronoi_polygons(LineString([(2, 2), (4, 2)]), only_edges=True)
    <LINESTRING (3 4, 3 0)>
    >>> voronoi_polygons(Point(2, 2))
    <GEOMETRYCOLLECTION EMPTY>
    """
    return lib.voronoi_polygons(geometry, tolerance, extend_to, only_edges, **kwargs)