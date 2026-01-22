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
def _add_transform_first(func):
    """
    A decorator that adds and validates the transform_first keyword argument.

    This handles a fast-path optimization that projects the points before
    creating any patches or lines. This means that the lines/patches will be
    calculated in projected-space, not data-space. It requires the first
    three arguments to be x, y, and z and all must be two-dimensional to use
    the fast-path option.

    This should be added after the _add_transform wrapper so that a transform
    is guaranteed to be present.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if kwargs.pop('transform_first', False):
            if len(args) < 3:
                raise ValueError('The X and Y arguments must be provided to use the transform_first=True fast-path.')
            x, y, z = (np.array(i) for i in args[:3])
            if not x.ndim == y.ndim == 2:
                raise ValueError('The X and Y arguments must be gridded 2-dimensional arrays')
            t = kwargs.pop('transform')
            pts = self.projection.transform_points(t, x, y)
            x = pts[..., 0].reshape(x.shape)
            y = pts[..., 1].reshape(y.shape)
            ind = np.argsort(x, axis=1)
            x = np.take_along_axis(x, ind, axis=1)
            y = np.take_along_axis(y, ind, axis=1)
            z = np.take_along_axis(z, ind, axis=1)
            args = (x, y, z) + args[3:]
        return func(self, *args, **kwargs)
    return wrapper