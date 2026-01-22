from pyproj.enums import BaseEnum
class VerticalCSAxis(BaseEnum):
    """
    .. versionadded:: 2.5.0

    Vertical Coordinate System Axis for creating axis with
    with :class:`pyproj.crs.coordinate_system.VerticalCS`

    Attributes
    ----------
    UP
    UP_FT
    UP_US_FT
    DEPTH
    DEPTH_FT
    DEPTH_US_FT
    GRAVITY_HEIGHT
    GRAVITY_HEIGHT_FT
    GRAVITY_HEIGHT_US_FT
    """
    GRAVITY_HEIGHT = 'GRAVITY_HEIGHT'
    GRAVITY_HEIGHT_FT = 'GRAVITY_HEIGHT_FT'
    GRAVITY_HEIGHT_US_FT = 'GRAVITY_HEIGHT_US_FT'
    DEPTH = 'DEPTH'
    DEPTH_FT = 'DEPTH_FT'
    DEPTH_US_FT = 'DEPTH_US_FT'
    UP = 'UP'
    UP_FT = 'UP_FT'
    UP_US_FT = 'UP_US_FT'