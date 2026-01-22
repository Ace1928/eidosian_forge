import warnings
from typing import Any
from pyproj._crs import CoordinateOperation
from pyproj.exceptions import CRSError
class PoleRotationNetCDFCFConversion(CoordinateOperation):
    """
    .. versionadded:: 3.3.0

    Class for constructing the Pole rotation (netCDF CF convention) conversion.

    http://cfconventions.org/cf-conventions/cf-conventions.html#_rotated_pole

    :ref:`PROJ docs <ob_tran>`
    """

    def __new__(cls, grid_north_pole_latitude: float, grid_north_pole_longitude: float, north_pole_grid_longitude: float=0.0):
        """
        Parameters
        ----------
        grid_north_pole_latitude: float
            Latitude of the North pole of the unrotated source CRS,
            expressed in the rotated geographic CRS (o_lat_p)
        grid_north_pole_longitude: float
            Longitude of projection center (lon_0 - 180).
        north_pole_grid_longitude: float, default=0.0
            Longitude of the North pole of the unrotated source CRS,
            expressed in the rotated geographic CRS (o_lon_p).
        """
        rot_latlon_json = {'$schema': 'https://proj.org/schemas/v0.4/projjson.schema.json', 'type': 'Conversion', 'name': 'Pole rotation (netCDF CF convention)', 'method': {'name': 'Pole rotation (netCDF CF convention)'}, 'parameters': [{'name': 'Grid north pole latitude (netCDF CF convention)', 'value': grid_north_pole_latitude, 'unit': 'degree'}, {'name': 'Grid north pole longitude (netCDF CF convention)', 'value': grid_north_pole_longitude, 'unit': 'degree'}, {'name': 'North pole grid longitude (netCDF CF convention)', 'value': north_pole_grid_longitude, 'unit': 'degree'}]}
        return cls.from_json_dict(rot_latlon_json)