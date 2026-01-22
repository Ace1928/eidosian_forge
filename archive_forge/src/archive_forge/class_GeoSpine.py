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
class GeoSpine(mspines.Spine):

    def __init__(self, axes, **kwargs):
        self._original_path = mpath.Path(np.empty((0, 2)))
        kwargs.setdefault('clip_on', False)
        super().__init__(axes, 'geo', self._original_path, **kwargs)

    def set_boundary(self, path, transform):
        self._original_path = path
        self.set_transform(transform)
        self.stale = True

    def _adjust_location(self):
        if self.stale:
            self._path = self._original_path.clip_to_bbox(self.axes.viewLim)
            self._path = mpath.Path(self._path.vertices, closed=True)

    def get_window_extent(self, renderer=None):
        self._adjust_location()
        return super().get_window_extent(renderer=renderer)

    @matplotlib.artist.allow_rasterization
    def draw(self, renderer):
        self._adjust_location()
        ret = super().draw(renderer)
        self.stale = False
        return ret

    def set_position(self, position):
        """GeoSpine does not support changing its position."""
        raise NotImplementedError('GeoSpine does not support changing its position.')