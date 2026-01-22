import os
from fiona.env import Env
from fiona._env import get_gdal_version_tuple
def _filter_supported_drivers():
    global supported_drivers
    with Env() as gdalenv:
        ogrdrv_names = gdalenv.drivers().keys()
        supported_drivers_copy = supported_drivers.copy()
        for drv in supported_drivers.keys():
            if drv not in ogrdrv_names:
                del supported_drivers_copy[drv]
    supported_drivers = supported_drivers_copy