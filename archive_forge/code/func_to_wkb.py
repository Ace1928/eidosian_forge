import numpy as np
from shapely import lib
from shapely._enum import ParamEnum
from shapely._ragged_array import from_ragged_array, to_ragged_array
from shapely.decorators import requires_geos
from shapely.errors import UnsupportedGEOSVersionError
def to_wkb(geometry, hex=False, output_dimension=3, byte_order=-1, include_srid=False, flavor='extended', **kwargs):
    """
    Converts to the Well-Known Binary (WKB) representation of a Geometry.

    The Well-Known Binary format is defined in the `OGC Simple Features
    Specification for SQL <https://www.opengeospatial.org/standards/sfs>`__.

    The following limitations apply to WKB serialization:

    - linearrings will be converted to linestrings
    - a point with only NaN coordinates is converted to an empty point
    - for GEOS <= 3.7, empty points are always serialized to 3D if
      output_dimension=3, and to 2D if output_dimension=2
    - for GEOS == 3.8, empty points are always serialized to 2D

    Parameters
    ----------
    geometry : Geometry or array_like
    hex : bool, default False
        If true, export the WKB as a hexidecimal string. The default is to
        return a binary bytes object.
    output_dimension : int, default 3
        The output dimension for the WKB. Supported values are 2 and 3.
        Specifying 3 means that up to 3 dimensions will be written but 2D
        geometries will still be represented as 2D in the WKB represenation.
    byte_order : int, default -1
        Defaults to native machine byte order (-1). Use 0 to force big endian
        and 1 for little endian.
    include_srid : bool, default False
        If True, the SRID is be included in WKB (this is an extension
        to the OGC WKB specification). Not allowed when flavor is "iso".
    flavor : {"iso", "extended"}, default "extended"
        Which flavor of WKB will be returned. The flavor determines how
        extra dimensionality is encoded with the type number, and whether
        SRID can be included in the WKB. ISO flavor is "more standard" for
        3D output, and does not support SRID embedding.
        Both flavors are equivalent when ``output_dimension=2`` (or with 2D
        geometries) and ``include_srid=False``.
        The `from_wkb` function can read both flavors.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> from shapely import Point
    >>> point = Point(1, 1)
    >>> to_wkb(point, byte_order=1)
    b'\\x01\\x01\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\xf0?\\x00\\x00\\x00\\x00\\x00\\x00\\xf0?'
    >>> to_wkb(point, hex=True, byte_order=1)
    '0101000000000000000000F03F000000000000F03F'
    """
    if not np.isscalar(hex):
        raise TypeError('hex only accepts scalar values')
    if not np.isscalar(output_dimension):
        raise TypeError('output_dimension only accepts scalar values')
    if not np.isscalar(byte_order):
        raise TypeError('byte_order only accepts scalar values')
    if not np.isscalar(include_srid):
        raise TypeError('include_srid only accepts scalar values')
    if not np.isscalar(flavor):
        raise TypeError('flavor only accepts scalar values')
    if lib.geos_version < (3, 10, 0) and flavor == 'iso':
        raise UnsupportedGEOSVersionError('The "iso" option requires at least GEOS 3.10.0')
    if flavor == 'iso' and include_srid:
        raise ValueError('flavor="iso" and include_srid=True cannot be used together')
    flavor = WKBFlavorOptions.get_value(flavor)
    return lib.to_wkb(geometry, np.bool_(hex), np.intc(output_dimension), np.intc(byte_order), np.bool_(include_srid), np.intc(flavor), **kwargs)