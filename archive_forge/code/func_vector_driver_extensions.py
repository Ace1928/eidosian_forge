import os
from fiona.env import Env
from fiona._env import get_gdal_version_tuple
def vector_driver_extensions():
    """
    Returns
    -------
    dict:
        Map of extensions to the driver.
    """
    from fiona.meta import extensions
    extension_to_driver = {}
    for drv, modes in supported_drivers.items():
        for extension in extensions(drv) or ():
            if 'w' in modes:
                extension_to_driver[extension] = extension_to_driver.get(extension, drv)
    return extension_to_driver