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
class TickLabels(AxisLabel):
    """
    Tick labels. While derived from `.Text`, this single artist draws all
    ticklabels. As in `.AxisLabel`, the position of the text is updated
    in the fly, so changing text position has no effect. Otherwise,
    the properties can be changed as a normal `.Text`. Unlike the
    ticklabels of the mainline Matplotlib, properties of a single
    ticklabel alone cannot be modified.

    To change the pad between ticks and ticklabels, use `~.AxisLabel.set_pad`.
    """

    def __init__(self, *, axis_direction='bottom', **kwargs):
        super().__init__(**kwargs)
        self.set_axis_direction(axis_direction)
        self._axislabel_pad = 0

    def get_ref_artist(self):
        return self._axis.get_ticklabels()[0]

    def set_axis_direction(self, label_direction):
        """
        Adjust the text angle and text alignment of ticklabels
        according to the Matplotlib convention.

        The *label_direction* must be one of [left, right, bottom, top].

        =====================    ========== ========= ========== ==========
        Property                 left       bottom    right      top
        =====================    ========== ========= ========== ==========
        ticklabel angle          90         0         -90        180
        ticklabel va             center     baseline  center     baseline
        ticklabel ha             right      center    right      center
        =====================    ========== ========= ========== ==========

        Note that the text angles are actually relative to (90 + angle
        of the direction to the ticklabel), which gives 0 for bottom
        axis.

        Parameters
        ----------
        label_direction : {"left", "bottom", "right", "top"}

        """
        self.set_default_alignment(label_direction)
        self.set_default_angle(label_direction)
        self._axis_direction = label_direction

    def invert_axis_direction(self):
        label_direction = self._get_opposite_direction(self._axis_direction)
        self.set_axis_direction(label_direction)

    def _get_ticklabels_offsets(self, renderer, label_direction):
        """
        Calculate the ticklabel offsets from the tick and their total heights.

        The offset only takes account the offset due to the vertical alignment
        of the ticklabels: if axis direction is bottom and va is 'top', it will
        return 0; if va is 'baseline', it will return (height-descent).
        """
        whd_list = self.get_texts_widths_heights_descents(renderer)
        if not whd_list:
            return (0, 0)
        r = 0
        va, ha = (self.get_va(), self.get_ha())
        if label_direction == 'left':
            pad = max((w for w, h, d in whd_list))
            if ha == 'left':
                r = pad
            elif ha == 'center':
                r = 0.5 * pad
        elif label_direction == 'right':
            pad = max((w for w, h, d in whd_list))
            if ha == 'right':
                r = pad
            elif ha == 'center':
                r = 0.5 * pad
        elif label_direction == 'bottom':
            pad = max((h for w, h, d in whd_list))
            if va == 'bottom':
                r = pad
            elif va == 'center':
                r = 0.5 * pad
            elif va == 'baseline':
                max_ascent = max((h - d for w, h, d in whd_list))
                max_descent = max((d for w, h, d in whd_list))
                r = max_ascent
                pad = max_ascent + max_descent
        elif label_direction == 'top':
            pad = max((h for w, h, d in whd_list))
            if va == 'top':
                r = pad
            elif va == 'center':
                r = 0.5 * pad
            elif va == 'baseline':
                max_ascent = max((h - d for w, h, d in whd_list))
                max_descent = max((d for w, h, d in whd_list))
                r = max_descent
                pad = max_ascent + max_descent
        return (r, pad)
    _default_alignments = dict(left=('center', 'right'), right=('center', 'left'), bottom=('baseline', 'center'), top=('baseline', 'center'))
    _default_angles = dict(left=90, right=-90, bottom=0, top=180)

    def draw(self, renderer):
        if not self.get_visible():
            self._axislabel_pad = self._external_pad
            return
        r, total_width = self._get_ticklabels_offsets(renderer, self._axis_direction)
        pad = self._external_pad + renderer.points_to_pixels(self.get_pad())
        self._offset_radius = r + pad
        for (x, y), a, l in self._locs_angles_labels:
            if not l.strip():
                continue
            self._ref_angle = a
            self.set_x(x)
            self.set_y(y)
            self.set_text(l)
            LabelBase.draw(self, renderer)
        self._axislabel_pad = total_width + pad

    def set_locs_angles_labels(self, locs_angles_labels):
        self._locs_angles_labels = locs_angles_labels

    def get_window_extents(self, renderer=None):
        if renderer is None:
            renderer = self.figure._get_renderer()
        if not self.get_visible():
            self._axislabel_pad = self._external_pad
            return []
        bboxes = []
        r, total_width = self._get_ticklabels_offsets(renderer, self._axis_direction)
        pad = self._external_pad + renderer.points_to_pixels(self.get_pad())
        self._offset_radius = r + pad
        for (x, y), a, l in self._locs_angles_labels:
            self._ref_angle = a
            self.set_x(x)
            self.set_y(y)
            self.set_text(l)
            bb = LabelBase.get_window_extent(self, renderer)
            bboxes.append(bb)
        self._axislabel_pad = total_width + pad
        return bboxes

    def get_texts_widths_heights_descents(self, renderer):
        """
        Return a list of ``(width, height, descent)`` tuples for ticklabels.

        Empty labels are left out.
        """
        whd_list = []
        for _loc, _angle, label in self._locs_angles_labels:
            if not label.strip():
                continue
            clean_line, ismath = self._preprocess_math(label)
            whd = renderer.get_text_width_height_descent(clean_line, self._fontproperties, ismath=ismath)
            whd_list.append(whd)
        return whd_list