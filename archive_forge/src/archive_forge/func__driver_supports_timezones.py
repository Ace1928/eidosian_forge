import os
from fiona.env import Env
from fiona._env import get_gdal_version_tuple
def _driver_supports_timezones(driver, field_type):
    """ Returns True if the driver supports timezones for field_type, False otherwise

        Note: this function is not part of Fiona's public API.
    """
    if field_type in _drivers_not_supporting_timezones and driver in _drivers_not_supporting_timezones[field_type]:
        if _drivers_not_supporting_timezones[field_type][driver] is None:
            return False
        elif _GDAL_VERSION < _drivers_not_supporting_timezones[field_type][driver]:
            return False
    return True