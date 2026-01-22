import numpy
from rasterio.env import GDALVersion
def _get_gdal_dtype(type_name):
    try:
        return dtype_rev[type_name]
    except KeyError:
        raise TypeError(f'Unsupported data type {type_name}. Allowed data types: {list(dtype_rev)}.')