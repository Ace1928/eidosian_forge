from pyproj.enums import BaseEnum
class CoordinateOperationType(BaseEnum):
    """
    .. versionadded:: 2.5.0

    Coordinate Operation Types for creating operation
    with :meth:`pyproj.crs.CoordinateOperation.from_name`

    Attributes
    ----------
    CONVERSION
    TRANSFORMATION
    CONCATENATED_OPERATION
    OTHER_COORDINATE_OPERATION
    """
    CONVERSION = 'CONVERSION'
    TRANSFORMATION = 'TRANSFORMATION'
    CONCATENATED_OPERATION = 'CONCATENATED_OPERATION'
    OTHER_COORDINATE_OPERATION = 'OTHER_COORDINATE_OPERATION'