import warnings
import numpy as np
from shapely import lib
from shapely.decorators import multithreading_enabled, requires_geos
def total_bounds(geometry, **kwargs):
    """Computes the total bounds (extent) of the geometry.

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Returns
    -------
    numpy ndarray of [xmin, ymin, xmax, ymax]

    Examples
    --------
    >>> from shapely import LineString, Point, Polygon
    >>> total_bounds(Point(2, 3)).tolist()
    [2.0, 3.0, 2.0, 3.0]
    >>> total_bounds([Point(2, 3), Point(4, 5)]).tolist()
    [2.0, 3.0, 4.0, 5.0]
    >>> total_bounds([
    ...     LineString([(0, 1), (0, 2), (3, 2)]),
    ...     LineString([(4, 4), (4, 6), (6, 7)])
    ... ]).tolist()
    [0.0, 1.0, 6.0, 7.0]
    >>> total_bounds(Polygon()).tolist()
    [nan, nan, nan, nan]
    >>> total_bounds([Polygon(), Point(2, 3)]).tolist()
    [2.0, 3.0, 2.0, 3.0]
    >>> total_bounds(None).tolist()
    [nan, nan, nan, nan]
    """
    b = bounds(geometry, **kwargs)
    if b.ndim == 1:
        return b
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.array([np.nanmin(b[..., 0]), np.nanmin(b[..., 1]), np.nanmax(b[..., 2]), np.nanmax(b[..., 3])])