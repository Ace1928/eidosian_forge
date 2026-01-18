import struct
from django.core.exceptions import ValidationError
from .const import (
def to_pgraster(rast):
    """
    Convert a GDALRaster into PostGIS Raster format.
    """
    rasterheader = (1, 0, len(rast.bands), rast.scale.x, rast.scale.y, rast.origin.x, rast.origin.y, rast.skew.x, rast.skew.y, rast.srs.srid, rast.width, rast.height)
    result = pack(POSTGIS_HEADER_STRUCTURE, rasterheader)
    for band in rast.bands:
        structure = 'B' + GDAL_TO_STRUCT[band.datatype()]
        pixeltype = GDAL_TO_POSTGIS[band.datatype()]
        if band.nodata_value is not None:
            pixeltype |= BANDTYPE_FLAG_HASNODATA
        bandheader = pack(structure, (pixeltype, band.nodata_value or 0))
        result += bandheader + band.data(as_memoryview=True)
    return result