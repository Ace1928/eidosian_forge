from the axis as some gridlines can never pass any axis.
import numpy as np
import matplotlib as mpl
from matplotlib import _api
import matplotlib.axes as maxes
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import mpl_axes
from .axisline_style import AxislineStyle  # noqa
from .axis_artist import AxisArtist, GridlinesCollection
class FixedAxisArtistHelperRectilinear(_FixedAxisArtistHelperBase):

    def __init__(self, axes, loc, nth_coord=None):
        """
        nth_coord = along which coordinate value varies
        in 2D, nth_coord = 0 ->  x axis, nth_coord = 1 -> y axis
        """
        super().__init__(loc, nth_coord)
        self.axis = [axes.xaxis, axes.yaxis][self.nth_coord]

    def get_tick_iterators(self, axes):
        """tick_loc, tick_angle, tick_label"""
        if self._loc in ['bottom', 'top']:
            angle_normal, angle_tangent = (90, 0)
        else:
            angle_normal, angle_tangent = (0, 90)
        major = self.axis.major
        major_locs = major.locator()
        major_labels = major.formatter.format_ticks(major_locs)
        minor = self.axis.minor
        minor_locs = minor.locator()
        minor_labels = minor.formatter.format_ticks(minor_locs)
        tick_to_axes = self.get_tick_transform(axes) - axes.transAxes

        def _f(locs, labels):
            for loc, label in zip(locs, labels):
                c = self._to_xy(loc, const=self._pos)
                c2 = tick_to_axes.transform(c)
                if mpl.transforms._interval_contains_close((0, 1), c2[self.nth_coord]):
                    yield (c, angle_normal, angle_tangent, label)
        return (_f(major_locs, major_labels), _f(minor_locs, minor_labels))