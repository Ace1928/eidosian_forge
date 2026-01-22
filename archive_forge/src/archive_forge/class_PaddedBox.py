import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
import matplotlib.artist as martist
import matplotlib.path as mpath
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms
from matplotlib.font_manager import FontProperties
from matplotlib.image import BboxImage
from matplotlib.patches import (
from matplotlib.transforms import Bbox, BboxBase, TransformedBbox
class PaddedBox(OffsetBox):
    """
    A container to add a padding around an `.Artist`.

    The `.PaddedBox` contains a `.FancyBboxPatch` that is used to visualize
    it when rendering.
    """

    def __init__(self, child, pad=0.0, *, draw_frame=False, patch_attrs=None):
        """
        Parameters
        ----------
        child : `~matplotlib.artist.Artist`
            The contained `.Artist`.
        pad : float, default: 0.0
            The padding in points. This will be scaled with the renderer dpi.
            In contrast, *width* and *height* are in *pixels* and thus not
            scaled.
        draw_frame : bool
            Whether to draw the contained `.FancyBboxPatch`.
        patch_attrs : dict or None
            Additional parameters passed to the contained `.FancyBboxPatch`.
        """
        super().__init__()
        self.pad = pad
        self._children = [child]
        self.patch = FancyBboxPatch(xy=(0.0, 0.0), width=1.0, height=1.0, facecolor='w', edgecolor='k', mutation_scale=1, snap=True, visible=draw_frame, boxstyle='square,pad=0')
        if patch_attrs is not None:
            self.patch.update(patch_attrs)

    def _get_bbox_and_child_offsets(self, renderer):
        pad = self.pad * renderer.points_to_pixels(1.0)
        return (self._children[0].get_bbox(renderer).padded(pad), [(0, 0)])

    def draw(self, renderer):
        bbox, offsets = self._get_bbox_and_child_offsets(renderer)
        px, py = self.get_offset(bbox, renderer)
        for c, (ox, oy) in zip(self.get_visible_children(), offsets):
            c.set_offset((px + ox, py + oy))
        self.draw_frame(renderer)
        for c in self.get_visible_children():
            c.draw(renderer)
        self.stale = False

    def update_frame(self, bbox, fontsize=None):
        self.patch.set_bounds(bbox.bounds)
        if fontsize:
            self.patch.set_mutation_scale(fontsize)
        self.stale = True

    def draw_frame(self, renderer):
        self.update_frame(self.get_window_extent(renderer))
        self.patch.draw(renderer)