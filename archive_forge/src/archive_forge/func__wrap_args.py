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
def _wrap_args(self, *args, **kwargs):
    """
        Handle the interpolation when a wrap could be involved with
        the data coordinates before passing on to Matplotlib.
        """
    default_shading = mpl.rcParams.get('pcolor.shading')
    if not (kwargs.get('shading', default_shading) in ('nearest', 'auto') and len(args) == 3 and getattr(kwargs.get('transform'), '_wrappable', False)):
        return (args, kwargs)
    kwargs['shading'] = 'flat'
    X = np.asanyarray(args[0])
    Y = np.asanyarray(args[1])
    nrows, ncols = np.asanyarray(args[2]).shape[:2]
    Nx = X.shape[-1]
    Ny = Y.shape[0]
    if X.ndim != 2 or X.shape[0] == 1:
        X = X.reshape(1, Nx).repeat(Ny, axis=0)
    if Y.ndim != 2 or Y.shape[1] == 1:
        Y = Y.reshape(Ny, 1).repeat(Nx, axis=1)

    def _interp_grid(X, wrap=0):
        if np.shape(X)[1] > 1:
            dX = np.diff(X, axis=1)
            if wrap:
                dX = (dX + wrap / 2) % wrap - wrap / 2
            dX = dX / 2
            X = np.hstack((X[:, [0]] - dX[:, [0]], X[:, :-1] + dX, X[:, [-1]] + dX[:, [-1]]))
        else:
            X = np.hstack((X, X))
        return X
    t = kwargs.get('transform')
    xwrap = abs(t.x_limits[1] - t.x_limits[0])
    if ncols == Nx:
        X = _interp_grid(X, wrap=xwrap)
        Y = _interp_grid(Y)
    if nrows == Ny:
        X = _interp_grid(X.T, wrap=xwrap).T
        Y = _interp_grid(Y.T).T
    return ((X, Y, args[2]), kwargs)