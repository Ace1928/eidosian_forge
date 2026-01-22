import math
import numpy as np
from contextlib import contextmanager
from matplotlib import (
from matplotlib.collections import (
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from . import proj3d
class Path3DCollection(PathCollection):
    """
    A collection of 3D paths.
    """

    def __init__(self, *args, zs=0, zdir='z', depthshade=True, **kwargs):
        """
        Create a collection of flat 3D paths with its normal vector
        pointed in *zdir* direction, and located at *zs* on the *zdir*
        axis. 'zs' can be a scalar or an array-like of the same length as
        the number of paths in the collection.

        Constructor arguments are the same as for
        :class:`~matplotlib.collections.PathCollection`. In addition,
        keywords *zs=0* and *zdir='z'* are available.

        Also, the keyword argument *depthshade* is available to indicate
        whether to shade the patches in order to give the appearance of depth
        (default is *True*). This is typically desired in scatter plots.
        """
        self._depthshade = depthshade
        self._in_draw = False
        super().__init__(*args, **kwargs)
        self.set_3d_properties(zs, zdir)
        self._offset_zordered = None

    def draw(self, renderer):
        with self._use_zordered_offset():
            with cbook._setattr_cm(self, _in_draw=True):
                super().draw(renderer)

    def set_sort_zpos(self, val):
        """Set the position to use for z-sorting."""
        self._sort_zpos = val
        self.stale = True

    def set_3d_properties(self, zs, zdir):
        """
        Set the *z* positions and direction of the paths.

        Parameters
        ----------
        zs : float or array of floats
            The location or locations to place the paths in the collection
            along the *zdir* axis.
        zdir : {'x', 'y', 'z'}
            Plane to plot paths orthogonal to.
            All paths must have the same direction.
            See `.get_dir_vector` for a description of the values.
        """
        self.update_scalarmappable()
        offsets = self.get_offsets()
        if len(offsets) > 0:
            xs, ys = offsets.T
        else:
            xs = []
            ys = []
        self._offsets3d = juggle_axes(xs, ys, np.atleast_1d(zs), zdir)
        self._sizes3d = self._sizes
        self._linewidths3d = np.array(self._linewidths)
        xs, ys, zs = self._offsets3d
        self._z_markers_idx = slice(-1)
        self._vzs = None
        self.stale = True

    def set_sizes(self, sizes, dpi=72.0):
        super().set_sizes(sizes, dpi)
        if not self._in_draw:
            self._sizes3d = sizes

    def set_linewidth(self, lw):
        super().set_linewidth(lw)
        if not self._in_draw:
            self._linewidths3d = np.array(self._linewidths)

    def get_depthshade(self):
        return self._depthshade

    def set_depthshade(self, depthshade):
        """
        Set whether depth shading is performed on collection members.

        Parameters
        ----------
        depthshade : bool
            Whether to shade the patches in order to give the appearance of
            depth.
        """
        self._depthshade = depthshade
        self.stale = True

    def do_3d_projection(self):
        xs, ys, zs = self._offsets3d
        vxs, vys, vzs, vis = proj3d.proj_transform_clip(xs, ys, zs, self.axes.M)
        z_markers_idx = self._z_markers_idx = np.argsort(vzs)[::-1]
        self._vzs = vzs
        if len(self._sizes3d) > 1:
            self._sizes = self._sizes3d[z_markers_idx]
        if len(self._linewidths3d) > 1:
            self._linewidths = self._linewidths3d[z_markers_idx]
        PathCollection.set_offsets(self, np.column_stack((vxs, vys)))
        vzs = vzs[z_markers_idx]
        vxs = vxs[z_markers_idx]
        vys = vys[z_markers_idx]
        self._offset_zordered = np.column_stack((vxs, vys))
        return np.min(vzs) if vzs.size else np.nan

    @contextmanager
    def _use_zordered_offset(self):
        if self._offset_zordered is None:
            yield
        else:
            old_offset = self._offsets
            super().set_offsets(self._offset_zordered)
            try:
                yield
            finally:
                self._offsets = old_offset

    def _maybe_depth_shade_and_sort_colors(self, color_array):
        color_array = _zalpha(color_array, self._vzs) if self._vzs is not None and self._depthshade else color_array
        if len(color_array) > 1:
            color_array = color_array[self._z_markers_idx]
        return mcolors.to_rgba_array(color_array, self._alpha)

    def get_facecolor(self):
        return self._maybe_depth_shade_and_sort_colors(super().get_facecolor())

    def get_edgecolor(self):
        if cbook._str_equal(self._edgecolors, 'face'):
            return self.get_facecolor()
        return self._maybe_depth_shade_and_sort_colors(super().get_edgecolor())