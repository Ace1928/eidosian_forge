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
class AnchoredOffsetbox(OffsetBox):
    """
    An offset box placed according to location *loc*.

    AnchoredOffsetbox has a single child.  When multiple children are needed,
    use an extra OffsetBox to enclose them.  By default, the offset box is
    anchored against its parent axes. You may explicitly specify the
    *bbox_to_anchor*.
    """
    zorder = 5
    codes = {'upper right': 1, 'upper left': 2, 'lower left': 3, 'lower right': 4, 'right': 5, 'center left': 6, 'center right': 7, 'lower center': 8, 'upper center': 9, 'center': 10}

    def __init__(self, loc, *, pad=0.4, borderpad=0.5, child=None, prop=None, frameon=True, bbox_to_anchor=None, bbox_transform=None, **kwargs):
        """
        Parameters
        ----------
        loc : str
            The box location.  Valid locations are
            'upper left', 'upper center', 'upper right',
            'center left', 'center', 'center right',
            'lower left', 'lower center', 'lower right'.
            For backward compatibility, numeric values are accepted as well.
            See the parameter *loc* of `.Legend` for details.
        pad : float, default: 0.4
            Padding around the child as fraction of the fontsize.
        borderpad : float, default: 0.5
            Padding between the offsetbox frame and the *bbox_to_anchor*.
        child : `.OffsetBox`
            The box that will be anchored.
        prop : `.FontProperties`
            This is only used as a reference for paddings. If not given,
            :rc:`legend.fontsize` is used.
        frameon : bool
            Whether to draw a frame around the box.
        bbox_to_anchor : `.BboxBase`, 2-tuple, or 4-tuple of floats
            Box that is used to position the legend in conjunction with *loc*.
        bbox_transform : None or :class:`matplotlib.transforms.Transform`
            The transform for the bounding box (*bbox_to_anchor*).
        **kwargs
            All other parameters are passed on to `.OffsetBox`.

        Notes
        -----
        See `.Legend` for a detailed description of the anchoring mechanism.
        """
        super().__init__(**kwargs)
        self.set_bbox_to_anchor(bbox_to_anchor, bbox_transform)
        self.set_child(child)
        if isinstance(loc, str):
            loc = _api.check_getitem(self.codes, loc=loc)
        self.loc = loc
        self.borderpad = borderpad
        self.pad = pad
        if prop is None:
            self.prop = FontProperties(size=mpl.rcParams['legend.fontsize'])
        else:
            self.prop = FontProperties._from_any(prop)
            if isinstance(prop, dict) and 'size' not in prop:
                self.prop.set_size(mpl.rcParams['legend.fontsize'])
        self.patch = FancyBboxPatch(xy=(0.0, 0.0), width=1.0, height=1.0, facecolor='w', edgecolor='k', mutation_scale=self.prop.get_size_in_points(), snap=True, visible=frameon, boxstyle='square,pad=0')

    def set_child(self, child):
        """Set the child to be anchored."""
        self._child = child
        if child is not None:
            child.axes = self.axes
        self.stale = True

    def get_child(self):
        """Return the child."""
        return self._child

    def get_children(self):
        """Return the list of children."""
        return [self._child]

    def get_bbox(self, renderer):
        fontsize = renderer.points_to_pixels(self.prop.get_size_in_points())
        pad = self.pad * fontsize
        return self.get_child().get_bbox(renderer).padded(pad)

    def get_bbox_to_anchor(self):
        """Return the bbox that the box is anchored to."""
        if self._bbox_to_anchor is None:
            return self.axes.bbox
        else:
            transform = self._bbox_to_anchor_transform
            if transform is None:
                return self._bbox_to_anchor
            else:
                return TransformedBbox(self._bbox_to_anchor, transform)

    def set_bbox_to_anchor(self, bbox, transform=None):
        """
        Set the bbox that the box is anchored to.

        *bbox* can be a Bbox instance, a list of [left, bottom, width,
        height], or a list of [left, bottom] where the width and
        height will be assumed to be zero. The bbox will be
        transformed to display coordinate by the given transform.
        """
        if bbox is None or isinstance(bbox, BboxBase):
            self._bbox_to_anchor = bbox
        else:
            try:
                l = len(bbox)
            except TypeError as err:
                raise ValueError(f'Invalid bbox: {bbox}') from err
            if l == 2:
                bbox = [bbox[0], bbox[1], 0, 0]
            self._bbox_to_anchor = Bbox.from_bounds(*bbox)
        self._bbox_to_anchor_transform = transform
        self.stale = True

    @_compat_get_offset
    def get_offset(self, bbox, renderer):
        pad = self.borderpad * renderer.points_to_pixels(self.prop.get_size_in_points())
        bbox_to_anchor = self.get_bbox_to_anchor()
        x0, y0 = _get_anchored_bbox(self.loc, Bbox.from_bounds(0, 0, bbox.width, bbox.height), bbox_to_anchor, pad)
        return (x0 - bbox.x0, y0 - bbox.y0)

    def update_frame(self, bbox, fontsize=None):
        self.patch.set_bounds(bbox.bounds)
        if fontsize:
            self.patch.set_mutation_scale(fontsize)

    def draw(self, renderer):
        if not self.get_visible():
            return
        bbox = self.get_window_extent(renderer)
        fontsize = renderer.points_to_pixels(self.prop.get_size_in_points())
        self.update_frame(bbox, fontsize)
        self.patch.draw(renderer)
        px, py = self.get_offset(self.get_bbox(renderer), renderer)
        self.get_child().set_offset((px, py))
        self.get_child().draw(renderer)
        self.stale = False