from the axis as some gridlines can never pass any axis.
import numpy as np
import matplotlib as mpl
from matplotlib import _api
import matplotlib.axes as maxes
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import mpl_axes
from .axisline_style import AxislineStyle  # noqa
from .axis_artist import AxisArtist, GridlinesCollection
class _FixedAxisArtistHelperBase(_AxisArtistHelperBase):
    """Helper class for a fixed (in the axes coordinate) axis."""
    passthru_pt = _api.deprecated('3.7')(property(lambda self: {'left': (0, 0), 'right': (1, 0), 'bottom': (0, 0), 'top': (0, 1)}[self._loc]))

    def __init__(self, loc, nth_coord=None):
        """``nth_coord = 0``: x-axis; ``nth_coord = 1``: y-axis."""
        self.nth_coord = nth_coord if nth_coord is not None else _api.check_getitem({'bottom': 0, 'top': 0, 'left': 1, 'right': 1}, loc=loc)
        if nth_coord == 0 and loc not in ['left', 'right'] or (nth_coord == 1 and loc not in ['bottom', 'top']):
            _api.warn_deprecated('3.7', message=f'loc={loc!r} is incompatible with {{nth_coord=}}; support is deprecated since %(since)s')
        self._loc = loc
        self._pos = {'bottom': 0, 'top': 1, 'left': 0, 'right': 1}[loc]
        super().__init__()
        self._path = Path(self._to_xy((0, 1), const=self._pos))

    def get_nth_coord(self):
        return self.nth_coord

    def get_line(self, axes):
        return self._path

    def get_line_transform(self, axes):
        return axes.transAxes

    def get_axislabel_transform(self, axes):
        return axes.transAxes

    def get_axislabel_pos_angle(self, axes):
        """
        Return the label reference position in transAxes.

        get_label_transform() returns a transform of (transAxes+offset)
        """
        return dict(left=((0.0, 0.5), 90), right=((1.0, 0.5), 90), bottom=((0.5, 0.0), 0), top=((0.5, 1.0), 0))[self._loc]

    def get_tick_transform(self, axes):
        return [axes.get_xaxis_transform(), axes.get_yaxis_transform()][self.nth_coord]