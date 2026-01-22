import warnings
from typing import Any
from pyproj._crs import CoordinateOperation
from pyproj.exceptions import CRSError
class VerticalPerspectiveConversion(CoordinateOperation):
    """
    .. versionadded:: 2.5.0

    Class for constructing the Vetical Perspective conversion.

    :ref:`PROJ docs <nsper>`
    """

    def __new__(cls, viewpoint_height: float, latitude_topocentric_origin: float=0.0, longitude_topocentric_origin: float=0.0, ellipsoidal_height_topocentric_origin: float=0.0, false_easting: float=0.0, false_northing: float=0.0):
        """
        Parameters
        ----------
        viewpoint_height: float
            Viewpoint height (h).
        latitude_topocentric_origin: float, default=0.0
            Latitude of topocentric origin (lat_0).
        longitude_topocentric_origin: float, default=0.0
            Longitude of topocentric origin (lon_0).
        ellipsoidal_height_topocentric_origin: float, default=0.0
            Ellipsoidal height of topocentric origin.
        false_easting: float, default=0.0
            False easting (x_0).
        false_northing: float, default=0.0
            False northing (y_0).

        """
        nsper_json = {'$schema': 'https://proj.org/schemas/v0.2/projjson.schema.json', 'type': 'Conversion', 'name': 'unknown', 'method': {'name': 'Vertical Perspective', 'id': {'authority': 'EPSG', 'code': 9838}}, 'parameters': [{'name': 'Latitude of topocentric origin', 'value': latitude_topocentric_origin, 'unit': 'degree', 'id': {'authority': 'EPSG', 'code': 8834}}, {'name': 'Longitude of topocentric origin', 'value': longitude_topocentric_origin, 'unit': 'degree', 'id': {'authority': 'EPSG', 'code': 8835}}, {'name': 'Ellipsoidal height of topocentric origin', 'value': ellipsoidal_height_topocentric_origin, 'unit': 'metre', 'id': {'authority': 'EPSG', 'code': 8836}}, {'name': 'Viewpoint height', 'value': viewpoint_height, 'unit': 'metre', 'id': {'authority': 'EPSG', 'code': 8840}}, {'name': 'False easting', 'value': false_easting, 'unit': 'metre', 'id': {'authority': 'EPSG', 'code': 8806}}, {'name': 'False northing', 'value': false_northing, 'unit': 'metre', 'id': {'authority': 'EPSG', 'code': 8807}}]}
        return cls.from_json_dict(nsper_json)