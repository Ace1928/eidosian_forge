from pyproj.enums import BaseEnum
class Ellipsoidal2DCSAxis(BaseEnum):
    """
    .. versionadded:: 2.5.0

    Ellipsoidal 2D Coordinate System Axis for creating axis with
    with :class:`pyproj.crs.coordinate_system.Ellipsoidal2DCS`

    Attributes
    ----------
    LONGITUDE_LATITUDE
    LATITUDE_LONGITUDE
    """
    LONGITUDE_LATITUDE = 'LONGITUDE_LATITUDE'
    LATITUDE_LONGITUDE = 'LATITUDE_LONGITUDE'