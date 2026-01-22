import warnings
from shapely.geometry.base import BaseGeometry
import pandas as pd
import numpy as np
from . import _compat as compat
from ._decorator import doc
class BaseSpatialIndex:

    @property
    def valid_query_predicates(self):
        """Returns valid predicates for this spatial index.

        Returns
        -------
        set
            Set of valid predicates for this spatial index.

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries([Point(0, 0), Point(1, 1)])
        >>> s.sindex.valid_query_predicates  # doctest: +SKIP
        {'contains', 'crosses', 'intersects', 'within', 'touches', 'overlaps', None, 'covers', 'contains_properly'}
        """
        raise NotImplementedError

    def query(self, geometry, predicate=None, sort=False):
        """
        Return the integer indices of all combinations of each input geometry
        and tree geometries where the bounding box of each input geometry
        intersects the bounding box of a tree geometry.

        If the input geometry is a scalar, this returns an array of shape (n, ) with
        the indices of the matching tree geometries.  If the input geometry is an
        array_like, this returns an array with shape (2,n) where the subarrays
        correspond to the indices of the input geometries and indices of the
        tree geometries associated with each.  To generate an array of pairs of
        input geometry index and tree geometry index, simply transpose the
        result.

        If a predicate is provided, the tree geometries are first queried based
        on the bounding box of the input geometry and then are further filtered
        to those that meet the predicate when comparing the input geometry to
        the tree geometry: ``predicate(geometry, tree_geometry)``.

        Bounding boxes are limited to two dimensions and are axis-aligned
        (equivalent to the ``bounds`` property of a geometry); any Z values
        present in input geometries are ignored when querying the tree.

        Any input geometry that is None or empty will never match geometries in
        the tree.

        Parameters
        ----------
        geometry : shapely.Geometry or array-like of geometries (numpy.ndarray, GeoSeries, GeometryArray)
            A single shapely geometry or array of geometries to query against
            the spatial index. For array-like, accepts both GeoPandas geometry
            iterables (GeoSeries, GeometryArray) or a numpy array of Shapely
            or PyGEOS geometries.
        predicate : {None, "contains", "contains_properly", "covered_by", "covers", "crosses", "intersects", "overlaps", "touches", "within"}, optional
            If predicate is provided, the input geometries are tested
            using the predicate function against each item in the tree
            whose extent intersects the envelope of the input geometry:
            ``predicate(input_geometry, tree_geometry)``.
            If possible, prepared geometries are used to help speed up the
            predicate operation.
        sort : bool, default False
            If True, the results will be sorted in ascending order. In case
            of 2D array, the result is sorted lexicographically using the
            geometries' indexes as the primary key and the sindex's indexes
            as the secondary key.
            If False, no additional sorting is applied (results are often
            sorted but there is no guarantee).

        Returns
        -------
        ndarray with shape (n,) if geometry is a scalar
            Integer indices for matching geometries from the spatial index
            tree geometries.

        OR

        ndarray with shape (2, n) if geometry is an array_like
            The first subarray contains input geometry integer indices.
            The second subarray contains tree geometry integer indices.

        Examples
        --------
        >>> from shapely.geometry import Point, box
        >>> s = geopandas.GeoSeries(geopandas.points_from_xy(range(10), range(10)))
        >>> s
        0    POINT (0.00000 0.00000)
        1    POINT (1.00000 1.00000)
        2    POINT (2.00000 2.00000)
        3    POINT (3.00000 3.00000)
        4    POINT (4.00000 4.00000)
        5    POINT (5.00000 5.00000)
        6    POINT (6.00000 6.00000)
        7    POINT (7.00000 7.00000)
        8    POINT (8.00000 8.00000)
        9    POINT (9.00000 9.00000)
        dtype: geometry

        Querying the tree with a scalar geometry:

        >>> s.sindex.query(box(1, 1, 3, 3))
        array([1, 2, 3])

        >>> s.sindex.query(box(1, 1, 3, 3), predicate="contains")
        array([2])

        Querying the tree with an array of geometries:

        >>> s2 = geopandas.GeoSeries([box(2, 2, 4, 4), box(5, 5, 6, 6)])
        >>> s2
        0    POLYGON ((4.00000 2.00000, 4.00000 4.00000, 2....
        1    POLYGON ((6.00000 5.00000, 6.00000 6.00000, 5....
        dtype: geometry

        >>> s.sindex.query(s2)
        array([[0, 0, 0, 1, 1],
               [2, 3, 4, 5, 6]])

        >>> s.sindex.query(s2, predicate="contains")
        array([[0],
               [3]])

        Notes
        -----
        In the context of a spatial join, input geometries are the "left"
        geometries that determine the order of the results, and tree geometries
        are "right" geometries that are joined against the left geometries. This
        effectively performs an inner join, where only those combinations of
        geometries that can be joined based on overlapping bounding boxes or
        optional predicate are returned.
        """
        raise NotImplementedError

    def query_bulk(self, geometry, predicate=None, sort=False):
        """
        DEPRECATED: use `query` instead.

        Returns all combinations of each input geometry and geometries in
        the tree where the envelope of each input geometry intersects with
        the envelope of a tree geometry.

        In the context of a spatial join, input geometries are the “left”
        geometries that determine the order of the results, and tree geometries
        are “right” geometries that are joined against the left geometries.
        This effectively performs an inner join, where only those combinations
        of geometries that can be joined based on envelope overlap or optional
        predicate are returned.

        When using the ``rtree`` package, this is not a vectorized function
        and may be slow. If speed is important, please use PyGEOS.

        Parameters
        ----------
        geometry : {GeoSeries, GeometryArray, numpy.array of PyGEOS geometries}
            Accepts GeoPandas geometry iterables (GeoSeries, GeometryArray)
            or a numpy array of PyGEOS geometries.
        predicate : {None, "contains", "contains_properly", "covered_by", "covers", "crosses", "intersects", "overlaps", "touches", "within"}, optional
            If predicate is provided, the input geometries are tested using
            the predicate function against each item in the tree whose extent
            intersects the envelope of the each input geometry:
            predicate(input_geometry, tree_geometry).  If possible, prepared
            geometries are used to help speed up the predicate operation.
        sort : bool, default False
            If True, results sorted lexicographically using
            geometry's indexes as the primary key and the sindex's indexes as the
            secondary key. If False, no additional sorting is applied.

        Returns
        -------
        ndarray with shape (2, n)
            The first subarray contains input geometry integer indexes.
            The second subarray contains tree geometry integer indexes.

        Examples
        --------
        >>> from shapely.geometry import Point, box
        >>> s = geopandas.GeoSeries(geopandas.points_from_xy(range(10), range(10)))
        >>> s
        0    POINT (0.00000 0.00000)
        1    POINT (1.00000 1.00000)
        2    POINT (2.00000 2.00000)
        3    POINT (3.00000 3.00000)
        4    POINT (4.00000 4.00000)
        5    POINT (5.00000 5.00000)
        6    POINT (6.00000 6.00000)
        7    POINT (7.00000 7.00000)
        8    POINT (8.00000 8.00000)
        9    POINT (9.00000 9.00000)
        dtype: geometry
        >>> s2 = geopandas.GeoSeries([box(2, 2, 4, 4), box(5, 5, 6, 6)])
        >>> s2
        0    POLYGON ((4.00000 2.00000, 4.00000 4.00000, 2....
        1    POLYGON ((6.00000 5.00000, 6.00000 6.00000, 5....
        dtype: geometry

        >>> s.sindex.query_bulk(s2)
        array([[0, 0, 0, 1, 1],
                [2, 3, 4, 5, 6]])

        >>> s.sindex.query_bulk(s2, predicate="contains")
        array([[0],
                [3]])
        """
        raise NotImplementedError

    def nearest(self, geometry, return_all=True, max_distance=None, return_distance=False, exclusive=False):
        """
        Return the nearest geometry in the tree for each input geometry in
        ``geometry``.

        .. note::
            ``nearest`` currently only works with PyGEOS >= 0.10.

            Note that if PyGEOS is not available, geopandas will use rtree
            for the spatial index, where nearest has a different
            function signature to temporarily preserve existing
            functionality. See the documentation of
            :meth:`rtree.index.Index.nearest` for the details on the
            ``rtree``-based implementation.

        If multiple tree geometries have the same distance from an input geometry,
        multiple results will be returned for that input geometry by default.
        Specify ``return_all=False`` to only get a single nearest geometry
        (non-deterministic which nearest is returned).

        In the context of a spatial join, input geometries are the "left"
        geometries that determine the order of the results, and tree geometries
        are "right" geometries that are joined against the left geometries.
        If ``max_distance`` is not set, this will effectively be a left join
        because every geometry in ``geometry`` will have a nearest geometry in
        the tree. However, if ``max_distance`` is used, this becomes an
        inner join, since some geometries in ``geometry`` may not have a match
        in the tree.

        For performance reasons, it is highly recommended that you set
        the ``max_distance`` parameter.

        Parameters
        ----------
        geometry : {shapely.geometry, GeoSeries, GeometryArray, numpy.array of PyGEOS geometries}
            A single shapely geometry, one of the GeoPandas geometry iterables
            (GeoSeries, GeometryArray), or a numpy array of PyGEOS geometries to query
            against the spatial index.
        return_all : bool, default True
            If there are multiple equidistant or intersecting nearest
            geometries, return all those geometries instead of a single
            nearest geometry.
        max_distance : float, optional
            Maximum distance within which to query for nearest items in tree.
            Must be greater than 0. By default None, indicating no distance limit.
        return_distance : bool, optional
            If True, will return distances in addition to indexes. By default False
        exclusive : bool, optional
            if True, the nearest geometries that are equal to the input geometry
            will not be returned. By default False.  Requires Shapely >= 2.0.

        Returns
        -------
        Indices or tuple of (indices, distances)
            Indices is an ndarray of shape (2,n) and distances (if present) an
            ndarray of shape (n).
            The first subarray of indices contains input geometry indices.
            The second subarray of indices contains tree geometry indices.

        Examples
        --------
        >>> from shapely.geometry import Point, box
        >>> s = geopandas.GeoSeries(geopandas.points_from_xy(range(10), range(10)))
        >>> s.head()
        0    POINT (0.00000 0.00000)
        1    POINT (1.00000 1.00000)
        2    POINT (2.00000 2.00000)
        3    POINT (3.00000 3.00000)
        4    POINT (4.00000 4.00000)
        dtype: geometry

        >>> s.sindex.nearest(Point(1, 1))
        array([[0],
               [1]])

        >>> s.sindex.nearest([box(4.9, 4.9, 5.1, 5.1)])
        array([[0],
               [5]])

        >>> s2 = geopandas.GeoSeries(geopandas.points_from_xy([7.6, 10], [7.6, 10]))
        >>> s2
        0    POINT (7.60000 7.60000)
        1    POINT (10.00000 10.00000)
        dtype: geometry

        >>> s.sindex.nearest(s2)
        array([[0, 1],
               [8, 9]])
        """
        raise NotImplementedError

    def intersection(self, coordinates):
        """Compatibility wrapper for rtree.index.Index.intersection,
        use ``query`` instead.

        Parameters
        ----------
        coordinates : sequence or array
            Sequence of the form (min_x, min_y, max_x, max_y)
            to query a rectangle or (x, y) to query a point.

        Examples
        --------
        >>> from shapely.geometry import Point, box
        >>> s = geopandas.GeoSeries(geopandas.points_from_xy(range(10), range(10)))
        >>> s
        0    POINT (0.00000 0.00000)
        1    POINT (1.00000 1.00000)
        2    POINT (2.00000 2.00000)
        3    POINT (3.00000 3.00000)
        4    POINT (4.00000 4.00000)
        5    POINT (5.00000 5.00000)
        6    POINT (6.00000 6.00000)
        7    POINT (7.00000 7.00000)
        8    POINT (8.00000 8.00000)
        9    POINT (9.00000 9.00000)
        dtype: geometry

        >>> s.sindex.intersection(box(1, 1, 3, 3).bounds)
        array([1, 2, 3])

        Alternatively, you can use ``query``:

        >>> s.sindex.query(box(1, 1, 3, 3))
        array([1, 2, 3])

        """
        raise NotImplementedError

    @property
    def size(self):
        """Size of the spatial index

        Number of leaves (input geometries) in the index.

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries(geopandas.points_from_xy(range(10), range(10)))
        >>> s
        0    POINT (0.00000 0.00000)
        1    POINT (1.00000 1.00000)
        2    POINT (2.00000 2.00000)
        3    POINT (3.00000 3.00000)
        4    POINT (4.00000 4.00000)
        5    POINT (5.00000 5.00000)
        6    POINT (6.00000 6.00000)
        7    POINT (7.00000 7.00000)
        8    POINT (8.00000 8.00000)
        9    POINT (9.00000 9.00000)
        dtype: geometry

        >>> s.sindex.size
        10
        """
        raise NotImplementedError

    @property
    def is_empty(self):
        """Check if the spatial index is empty

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries(geopandas.points_from_xy(range(10), range(10)))
        >>> s
        0    POINT (0.00000 0.00000)
        1    POINT (1.00000 1.00000)
        2    POINT (2.00000 2.00000)
        3    POINT (3.00000 3.00000)
        4    POINT (4.00000 4.00000)
        5    POINT (5.00000 5.00000)
        6    POINT (6.00000 6.00000)
        7    POINT (7.00000 7.00000)
        8    POINT (8.00000 8.00000)
        9    POINT (9.00000 9.00000)
        dtype: geometry

        >>> s.sindex.is_empty
        False

        >>> s2 = geopandas.GeoSeries()
        >>> s2.sindex.is_empty
        True
        """
        raise NotImplementedError