from matplotlib.collections import QuadMesh
import numpy as np
import numpy.ma as ma
from cartopy.mpl import _MPL_38
def _split_wrapped_mesh_data(C, mask):
    """
    Helper function for splitting GeoQuadMesh array values between the
    pcolormesh and pcolor objects when wrapping.  Apply a mask to the grid
    cells that should not be plotted with each method.

    """
    C_mask = getattr(C, 'mask', None)
    if C.ndim == 3:
        if not _MPL_38:
            raise ValueError('GeoQuadMesh wrapping for RGB(A) requires Matplotlib v3.8 or later')
        mask = np.broadcast_to(mask[..., np.newaxis], C.shape)
    full_mask = mask if C_mask is None else mask | C_mask
    pcolormesh_data = ma.array(C, mask=full_mask)
    full_mask = ~mask if C_mask is None else ~mask | C_mask
    pcolor_data = ma.array(C, mask=full_mask)
    return (pcolormesh_data, pcolor_data, ~mask)