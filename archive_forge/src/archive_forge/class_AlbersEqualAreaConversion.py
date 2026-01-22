import warnings
from typing import Any
from pyproj._crs import CoordinateOperation
from pyproj.exceptions import CRSError
class AlbersEqualAreaConversion(CoordinateOperation):
    """
    .. versionadded:: 2.5.0

    Class for constructing the Albers Equal Area Conversion.

    :ref:`PROJ docs <aea>`
    """

    def __new__(cls, latitude_first_parallel: float, latitude_second_parallel: float, latitude_false_origin: float=0.0, longitude_false_origin: float=0.0, easting_false_origin: float=0.0, northing_false_origin: float=0.0):
        """
        Parameters
        ----------
        latitude_first_parallel: float
            First standard parallel (lat_1).
        latitude_second_parallel: float
            Second standard parallel (lat_2).
        latitude_false_origin: float, default=0.0
            Latitude of projection center (lat_0).
        longitude_false_origin: float, default=0.0
            Longitude of projection center (lon_0).
        easting_false_origin: float, default=0.0
            False easting (x_0).
        northing_false_origin: float, default=0.0
            False northing (y_0).
        """
        aea_json = {'$schema': 'https://proj.org/schemas/v0.2/projjson.schema.json', 'type': 'Conversion', 'name': 'unknown', 'method': {'name': 'Albers Equal Area', 'id': {'authority': 'EPSG', 'code': 9822}}, 'parameters': [{'name': 'Latitude of false origin', 'value': latitude_false_origin, 'unit': 'degree', 'id': {'authority': 'EPSG', 'code': 8821}}, {'name': 'Longitude of false origin', 'value': longitude_false_origin, 'unit': 'degree', 'id': {'authority': 'EPSG', 'code': 8822}}, {'name': 'Latitude of 1st standard parallel', 'value': latitude_first_parallel, 'unit': 'degree', 'id': {'authority': 'EPSG', 'code': 8823}}, {'name': 'Latitude of 2nd standard parallel', 'value': latitude_second_parallel, 'unit': 'degree', 'id': {'authority': 'EPSG', 'code': 8824}}, {'name': 'Easting at false origin', 'value': easting_false_origin, 'unit': {'type': 'LinearUnit', 'name': 'Metre', 'conversion_factor': 1}, 'id': {'authority': 'EPSG', 'code': 8826}}, {'name': 'Northing at false origin', 'value': northing_false_origin, 'unit': {'type': 'LinearUnit', 'name': 'Metre', 'conversion_factor': 1}, 'id': {'authority': 'EPSG', 'code': 8827}}]}
        return cls.from_json_dict(aea_json)