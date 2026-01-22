from click import FileError
class DatasetAttributeError(RasterioError, NotImplementedError):
    """Raised when dataset attributes are misused"""