import warnings
from typing import Any
from pyproj._crs import CoordinateOperation
from pyproj.exceptions import CRSError
class SinusoidalConversion(CoordinateOperation):
    """
    .. versionadded:: 2.5.0

    Class for constructing the Sinusoidal conversion.

    :ref:`PROJ docs <sinu>`
    """

    def __new__(cls, longitude_natural_origin: float=0.0, false_easting: float=0.0, false_northing: float=0.0):
        """
        Parameters
        ----------
        longitude_natural_origin: float, default=0.0
            Longitude of projection center (lon_0).
        false_easting: float, default=0.0
            False easting (x_0).
        false_northing: float, default=0.0
            False northing (y_0).

        """
        sinu_json = {'$schema': 'https://proj.org/schemas/v0.2/projjson.schema.json', 'type': 'Conversion', 'name': 'unknown', 'method': {'name': 'Sinusoidal'}, 'parameters': [{'name': 'Longitude of natural origin', 'value': longitude_natural_origin, 'unit': 'degree', 'id': {'authority': 'EPSG', 'code': 8802}}, {'name': 'False easting', 'value': false_easting, 'unit': 'metre', 'id': {'authority': 'EPSG', 'code': 8806}}, {'name': 'False northing', 'value': false_northing, 'unit': 'metre', 'id': {'authority': 'EPSG', 'code': 8807}}]}
        return cls.from_json_dict(sinu_json)