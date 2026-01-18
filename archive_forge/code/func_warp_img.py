import numpy as np
import cartopy.crs as ccrs
def warp_img(fname, target_proj, source_proj=None, target_res=(400, 200)):
    """
    Regrid the image file from the source projection to the target projection.

    Parameters
    ----------
    fname
        Image filename to be loaded and warped.
    target_proj
        The target :class:`~cartopy.crs.Projection` instance for the image.
    source_proj: optional
        The source :class:`~cartopy.crs.Projection` instance of the image.
        Defaults to a :class:`~cartopy.crs.PlateCarree` projection.
    target_res: optional
        The (nx, ny) resolution of the target projection. Where nx defaults to
        400 sample points, and ny defaults to 200 sample points.

    """
    if source_proj is None:
        source_proj = ccrs.PlateCarree()
    raise NotImplementedError('Not yet implemented.')