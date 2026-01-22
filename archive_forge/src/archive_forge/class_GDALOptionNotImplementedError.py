from click import FileError
class GDALOptionNotImplementedError(RasterioError):
    """A dataset opening or dataset creation option can't be supported

    This will be raised from Rasterio's shim modules. For example, when
    a user passes arguments to open_dataset() that can't be evaluated
    by GDAL 1.x.
    """