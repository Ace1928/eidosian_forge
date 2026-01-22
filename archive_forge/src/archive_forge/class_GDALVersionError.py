from click import FileError
class GDALVersionError(RasterioError):
    """Raised if the runtime version of GDAL does not meet the required version of GDAL."""