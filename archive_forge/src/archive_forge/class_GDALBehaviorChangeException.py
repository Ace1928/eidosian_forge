from click import FileError
class GDALBehaviorChangeException(RuntimeError):
    """Raised when GDAL's behavior differs from the given arguments.  For
    example, antimeridian cutting is always on as of GDAL 2.2.0.  Users
    expecting it to be off will be presented with a MultiPolygon when the
    rest of their code expects a Polygon.

    Examples
    --------

    .. code-block:: python

        # Raises an exception on GDAL >= 2.2.0
        rasterio.warp.transform_geometry(
            src_crs, dst_crs, antimeridian_cutting=False)
    """