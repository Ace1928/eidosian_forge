import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.patches as mpatches
from matplotlib.path import Path
from mpl_toolkits.axes_grid1.parasite_axes import host_axes_class_factory
from . import axislines, grid_helper_curvelinear
from .axis_artist import AxisArtist
from .grid_finder import ExtremeFinderSimple
class FixedAxisArtistHelper(grid_helper_curvelinear.FloatingAxisArtistHelper):

    def __init__(self, grid_helper, side, nth_coord_ticks=None):
        """
        nth_coord = along which coordinate value varies.
         nth_coord = 0 ->  x axis, nth_coord = 1 -> y axis
        """
        lon1, lon2, lat1, lat2 = grid_helper.grid_finder.extreme_finder(*[None] * 5)
        value, nth_coord = _api.check_getitem(dict(left=(lon1, 0), right=(lon2, 0), bottom=(lat1, 1), top=(lat2, 1)), side=side)
        super().__init__(grid_helper, nth_coord, value, axis_direction=side)
        if nth_coord_ticks is None:
            nth_coord_ticks = nth_coord
        self.nth_coord_ticks = nth_coord_ticks
        self.value = value
        self.grid_helper = grid_helper
        self._side = side

    def update_lim(self, axes):
        self.grid_helper.update_lim(axes)
        self._grid_info = self.grid_helper._grid_info

    def get_tick_iterators(self, axes):
        """tick_loc, tick_angle, tick_label, (optionally) tick_label"""
        grid_finder = self.grid_helper.grid_finder
        lat_levs, lat_n, lat_factor = self._grid_info['lat_info']
        yy0 = lat_levs / lat_factor
        lon_levs, lon_n, lon_factor = self._grid_info['lon_info']
        xx0 = lon_levs / lon_factor
        extremes = self.grid_helper.grid_finder.extreme_finder(*[None] * 5)
        xmin, xmax = sorted(extremes[:2])
        ymin, ymax = sorted(extremes[2:])

        def trf_xy(x, y):
            trf = grid_finder.get_transform() + axes.transData
            return trf.transform(np.column_stack(np.broadcast_arrays(x, y))).T
        if self.nth_coord == 0:
            mask = (ymin <= yy0) & (yy0 <= ymax)
            (xx1, yy1), (dxx1, dyy1), (dxx2, dyy2) = grid_helper_curvelinear._value_and_jacobian(trf_xy, self.value, yy0[mask], (xmin, xmax), (ymin, ymax))
            labels = self._grid_info['lat_labels']
        elif self.nth_coord == 1:
            mask = (xmin <= xx0) & (xx0 <= xmax)
            (xx1, yy1), (dxx2, dyy2), (dxx1, dyy1) = grid_helper_curvelinear._value_and_jacobian(trf_xy, xx0[mask], self.value, (xmin, xmax), (ymin, ymax))
            labels = self._grid_info['lon_labels']
        labels = [l for l, m in zip(labels, mask) if m]
        angle_normal = np.arctan2(dyy1, dxx1)
        angle_tangent = np.arctan2(dyy2, dxx2)
        mm = (dyy1 == 0) & (dxx1 == 0)
        angle_normal[mm] = angle_tangent[mm] + np.pi / 2
        tick_to_axes = self.get_tick_transform(axes) - axes.transAxes
        in_01 = functools.partial(mpl.transforms._interval_contains_close, (0, 1))

        def f1():
            for x, y, normal, tangent, lab in zip(xx1, yy1, angle_normal, angle_tangent, labels):
                c2 = tick_to_axes.transform((x, y))
                if in_01(c2[0]) and in_01(c2[1]):
                    yield ([x, y], *np.rad2deg([normal, tangent]), lab)
        return (f1(), iter([]))

    def get_line(self, axes):
        self.update_lim(axes)
        k, v = dict(left=('lon_lines0', 0), right=('lon_lines0', 1), bottom=('lat_lines0', 0), top=('lat_lines0', 1))[self._side]
        xx, yy = self._grid_info[k][v]
        return Path(np.column_stack([xx, yy]))