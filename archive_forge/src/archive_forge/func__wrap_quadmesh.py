import collections
import contextlib
import functools
import json
import os
from pathlib import Path
import warnings
import weakref
import matplotlib as mpl
import matplotlib.artist
import matplotlib.axes
import matplotlib.contour
from matplotlib.image import imread
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.spines as mspines
import matplotlib.transforms as mtransforms
import numpy as np
import numpy.ma as ma
import shapely.geometry as sgeom
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.mpl import _MPL_38
import cartopy.mpl.contour
import cartopy.mpl.feature_artist as feature_artist
import cartopy.mpl.geocollection
import cartopy.mpl.patch as cpatch
from cartopy.mpl.slippy_image_artist import SlippyImageArtist
def _wrap_quadmesh(self, collection, **kwargs):
    """
        Handles the Quadmesh collection when any of the quadrilaterals
        cross the boundary of the projection.
        """
    t = kwargs.get('transform', None)
    coords = collection._coordinates
    Ny, Nx, _ = coords.shape
    if kwargs.get('shading') == 'gouraud':
        data_shape = (Ny, Nx, -1)
    else:
        data_shape = (Ny - 1, Nx - 1, -1)
    C = collection.get_array().reshape(data_shape)
    if C.shape[-1] == 1:
        C = C.squeeze(axis=-1)
    transformed_pts = self.projection.transform_points(t, coords[..., 0], coords[..., 1])
    with np.errstate(invalid='ignore'):
        xs, ys = (transformed_pts[..., 0], transformed_pts[..., 1])
        diagonal0_lengths = np.hypot(xs[1:, 1:] - xs[:-1, :-1], ys[1:, 1:] - ys[:-1, :-1])
        diagonal1_lengths = np.hypot(xs[1:, :-1] - xs[:-1, 1:], ys[1:, :-1] - ys[:-1, 1:])
        size_limit = abs(self.projection.x_limits[1] - self.projection.x_limits[0]) / (2 * np.sqrt(2))
        mask = np.isnan(diagonal0_lengths) | (diagonal0_lengths > size_limit) | np.isnan(diagonal1_lengths) | (diagonal1_lengths > size_limit)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN slice encountered')
        corners = ((np.nanmin(xs), np.nanmin(ys)), (np.nanmax(xs), np.nanmax(ys)))
    collection._corners = mtransforms.Bbox(corners)
    self.update_datalim(collection._corners)
    if not (getattr(t, '_wrappable', False) and getattr(self.projection, '_wrappable', False)) or not np.any(mask):
        return collection
    if kwargs.get('shading') == 'gouraud':
        warnings.warn('Handling wrapped coordinates with gouraud shading is likely to introduce artifacts. It is recommended to remove the wrap manually before calling pcolormesh.')
        gmask = np.zeros(data_shape, dtype=bool)
        gmask[:-1, :-1] |= mask
        gmask[1:, :-1] |= mask
        gmask[1:, 1:] |= mask
        gmask[:-1, 1:] |= mask
        mask = gmask
    if collection.get_cmap()._rgba_bad[3] != 0.0:
        warnings.warn("The colormap's 'bad' has been set, but in order to wrap pcolormesh across the map it must be fully transparent.", stacklevel=3)
    pcolormesh_data, pcolor_data, pcolor_mask = cartopy.mpl.geocollection._split_wrapped_mesh_data(C, mask)
    collection.set_array(pcolormesh_data)
    zorder = collection.zorder - 0.1
    kwargs.pop('zorder', None)
    kwargs.pop('shading', None)
    kwargs.setdefault('snap', False)
    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    norm = kwargs.pop('norm', None)
    cmap = kwargs.pop('cmap', None)
    if not _MPL_38:
        pcolor_zeros = np.ma.array(np.zeros(C.shape), mask=pcolor_mask)
        pcolor_col = self.pcolor(coords[..., 0], coords[..., 1], pcolor_zeros, zorder=zorder, **kwargs)
        pcolor_col.set_array(pcolor_data[mask].ravel())
    else:
        pcolor_col = self.pcolor(coords[..., 0], coords[..., 1], pcolor_data, zorder=zorder, **kwargs)
        pcolor_col.set_array(pcolor_data)
    pcolor_col.set_cmap(cmap)
    pcolor_col.set_norm(norm)
    pcolor_col.set_clim(vmin, vmax)
    pcolor_col.norm.autoscale_None(C)
    collection._wrapped_mask = mask
    collection._wrapped_collection_fix = pcolor_col
    return collection