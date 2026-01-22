import importlib
import os
import platform
import sys
def _get_gdal_info():
    """Information on system GDAL

    Returns
    -------
    dict:
        system GDAL information
    """
    import rasterio
    blob = [('rasterio', rasterio.__version__), ('GDAL', rasterio.__gdal_version__), ('PROJ', rasterio.__proj_version__), ('GEOS', rasterio.__geos_version__), ('PROJ DATA', os.pathsep.join(rasterio._env.get_proj_data_search_paths())), ('GDAL DATA', rasterio._env.get_gdal_data())]
    return dict(blob)