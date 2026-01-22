import numpy
from rasterio.env import GDALVersion
def _is_complex_int(dtype):
    return isinstance(dtype, str) and dtype.startswith('complex_int')