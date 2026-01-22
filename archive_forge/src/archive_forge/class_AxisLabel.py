from operator import methodcaller
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.text as mtext
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import (
from .axisline_style import AxislineStyle
class AxisLabel(AttributeCopier, LabelBase):
    """
    Axis label. Derived from `.Text`. The position of the text is updated
    in the fly, so changing text position has no effect. Otherwise, the
    properties can be changed as a normal `.Text`.

    To change the pad between tick labels and axis label, use `set_pad`.
    """

    def __init__(self, *args, axis_direction='bottom', axis=None, **kwargs):
        self._axis = axis
        self._pad = 5
        self._external_pad = 0
        LabelBase.__init__(self, *args, **kwargs)
        self.set_axis_direction(axis_direction)

    def set_pad(self, pad):
        """
        Set the internal pad in points.

        The actual pad will be the sum of the internal pad and the
        external pad (the latter is set automatically by the `.AxisArtist`).

        Parameters
        ----------
        pad : float
            The internal pad in points.
        """
        self._pad = pad

    def get_pad(self):
        """
        Return the internal pad in points.

        See `.set_pad` for more details.
        """
        return self._pad

    def get_ref_artist(self):
        return self._axis.get_label()

    def get_text(self):
        t = super().get_text()
        if t == '__from_axes__':
            return self._axis.get_label().get_text()
        return self._text
    _default_alignments = dict(left=('bottom', 'center'), right=('top', 'center'), bottom=('top', 'center'), top=('bottom', 'center'))

    def set_default_alignment(self, d):
        """
        Set the default alignment. See `set_axis_direction` for details.

        Parameters
        ----------
        d : {"left", "bottom", "right", "top"}
        """
        va, ha = _api.check_getitem(self._default_alignments, d=d)
        self.set_va(va)
        self.set_ha(ha)
    _default_angles = dict(left=180, right=0, bottom=0, top=180)

    def set_default_angle(self, d):
        """
        Set the default angle. See `set_axis_direction` for details.

        Parameters
        ----------
        d : {"left", "bottom", "right", "top"}
        """
        self.set_rotation(_api.check_getitem(self._default_angles, d=d))

    def set_axis_direction(self, d):
        """
        Adjust the text angle and text alignment of axis label
        according to the matplotlib convention.

        =====================    ========== ========= ========== ==========
        Property                 left       bottom    right      top
        =====================    ========== ========= ========== ==========
        axislabel angle          180        0         0          180
        axislabel va             center     top       center     bottom
        axislabel ha             right      center    right      center
        =====================    ========== ========= ========== ==========

        Note that the text angles are actually relative to (90 + angle
        of the direction to the ticklabel), which gives 0 for bottom
        axis.

        Parameters
        ----------
        d : {"left", "bottom", "right", "top"}
        """
        self.set_default_alignment(d)
        self.set_default_angle(d)

    def get_color(self):
        return self.get_attribute_from_ref_artist('color')

    def draw(self, renderer):
        if not self.get_visible():
            return
        self._offset_radius = self._external_pad + renderer.points_to_pixels(self.get_pad())
        super().draw(renderer)

    def get_window_extent(self, renderer=None):
        if renderer is None:
            renderer = self.figure._get_renderer()
        if not self.get_visible():
            return
        r = self._external_pad + renderer.points_to_pixels(self.get_pad())
        self._offset_radius = r
        bb = super().get_window_extent(renderer)
        return bb