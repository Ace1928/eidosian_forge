from click import FileError
class DriverCapabilityError(RasterioError, ValueError):
    """Raised when a format driver can't a feature such as writing."""