import numpy as np
from shapely import lib
from shapely._enum import ParamEnum
from shapely._ragged_array import from_ragged_array, to_ragged_array
from shapely.decorators import requires_geos
from shapely.errors import UnsupportedGEOSVersionError
@requires_geos('3.10.0')
def to_geojson(geometry, indent=None, **kwargs):
    """Converts to the GeoJSON representation of a Geometry.

    The GeoJSON format is defined in the `RFC 7946 <https://geojson.org/>`__.
    NaN (not-a-number) coordinates will be written as 'null'.

    The following are currently unsupported:

    - Geometries of type LINEARRING: these are output as 'null'.
    - Three-dimensional geometries: the third dimension is ignored.

    Parameters
    ----------
    geometry : str, bytes or array_like
    indent : int, optional
        If indent is a non-negative integer, then GeoJSON will be formatted.
        An indent level of 0 will only insert newlines. None (the default)
        selects the most compact representation.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> from shapely import Point
    >>> point = Point(1, 1)
    >>> to_geojson(point)
    '{"type":"Point","coordinates":[1.0,1.0]}'
    >>> print(to_geojson(point, indent=2))
    {
      "type": "Point",
      "coordinates": [
          1.0,
          1.0
      ]
    }
    """
    if indent is None:
        indent = -1
    elif not np.isscalar(indent):
        raise TypeError('indent only accepts scalar values')
    elif indent < 0:
        raise ValueError('indent cannot be negative')
    return lib.to_geojson(geometry, np.intc(indent), **kwargs)