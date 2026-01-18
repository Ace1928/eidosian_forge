import numpy as np
import cartopy.crs as ccrs
def warp_array(array, target_proj, source_proj=None, target_res=(400, 200), source_extent=None, target_extent=None, mask_extrapolated=False):
    """
    Regrid the data array from the source projection to the target projection.

    Also see, :func:`~cartopy.img_transform.regrid`.

    Parameters
    ----------
    array
        The :class:`numpy.ndarray` of data to be regridded to the target
        projection.
    target_proj
        The target :class:`~cartopy.crs.Projection` instance for the data.
    source_proj: optional
        The source :class:`~cartopy.crs.Projection' instance of the data.
        Defaults to a :class:`~cartopy.crs.PlateCarree` projection.
    target_res: optional
        The (nx, ny) resolution of the target projection. Where nx defaults to
        400 sample points, and ny defaults to 200 sample points.
    source_extent: optional
        The (x-lower, x-upper, y-lower, y-upper) extent in native
        source projection coordinates.
    target_extent: optional
        The (x-lower, x-upper, y-lower, y-upper) extent in native
        target projection coordinates.
    mask_extrapolated: optional
        Assume that the source coordinate is rectilinear and so mask the
        resulting target grid values which lie outside the source grid
        domain.

    Returns
    -------
    array, extent
        A tuple of the regridded :class:`numpy.ndarray` in the target
        projection and the (x-lower, x-upper, y-lower, y-upper) target
        projection extent.

    """
    if source_extent is None:
        source_extent = [None] * 4
    if target_extent is None:
        target_extent = [None] * 4
    source_x_extents = source_extent[:2]
    source_y_extents = source_extent[2:]
    target_x_extents = target_extent[:2]
    target_y_extents = target_extent[2:]
    if source_proj is None:
        source_proj = ccrs.PlateCarree()
    ny, nx = array.shape[:2]
    source_native_xy = mesh_projection(source_proj, nx, ny, x_extents=source_x_extents, y_extents=source_y_extents)
    target_native_x, target_native_y, extent = mesh_projection(target_proj, target_res[0], target_res[1], x_extents=target_x_extents, y_extents=target_y_extents)
    array = regrid(array, source_native_xy[0], source_native_xy[1], source_proj, target_proj, target_native_x, target_native_y, mask_extrapolated)
    return (array, extent)