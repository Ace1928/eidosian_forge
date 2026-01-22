from click import FileError
class DatasetIOShapeError(RasterioError):
    """Raised when data buffer shape is a mismatch when reading and writing"""