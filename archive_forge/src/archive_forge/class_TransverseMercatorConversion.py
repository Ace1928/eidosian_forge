import warnings
from typing import Any
from pyproj._crs import CoordinateOperation
from pyproj.exceptions import CRSError
class TransverseMercatorConversion(CoordinateOperation):
    """
    .. versionadded:: 2.5.0

    Class for constructing the Transverse Mercator conversion.

    :ref:`PROJ docs <tmerc>`
    """

    def __new__(cls, latitude_natural_origin: float=0.0, longitude_natural_origin: float=0.0, false_easting: float=0.0, false_northing: float=0.0, scale_factor_natural_origin: float=1.0):
        """
        Parameters
        ----------
        latitude_natural_origin: float, default=0.0
            Latitude of projection center (lat_0).
        longitude_natural_origin: float, default=0.0
            Longitude of projection center (lon_0).
        false_easting: float, default=0.0
            False easting (x_0).
        false_northing: float, default=0.0
            False northing (y_0).
        scale_factor_natural_origin: float, default=1.0
            Scale factor at natural origin (k or k_0).

        """
        tmerc_json = {'$schema': 'https://proj.org/schemas/v0.2/projjson.schema.json', 'type': 'Conversion', 'name': 'unknown', 'method': {'name': 'Transverse Mercator', 'id': {'authority': 'EPSG', 'code': 9807}}, 'parameters': [{'name': 'Latitude of natural origin', 'value': latitude_natural_origin, 'unit': 'degree', 'id': {'authority': 'EPSG', 'code': 8801}}, {'name': 'Longitude of natural origin', 'value': longitude_natural_origin, 'unit': 'degree', 'id': {'authority': 'EPSG', 'code': 8802}}, {'name': 'Scale factor at natural origin', 'value': scale_factor_natural_origin, 'unit': 'unity', 'id': {'authority': 'EPSG', 'code': 8805}}, {'name': 'False easting', 'value': false_easting, 'unit': 'metre', 'id': {'authority': 'EPSG', 'code': 8806}}, {'name': 'False northing', 'value': false_northing, 'unit': 'metre', 'id': {'authority': 'EPSG', 'code': 8807}}]}
        return cls.from_json_dict(tmerc_json)