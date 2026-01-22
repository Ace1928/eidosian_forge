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
class VPacker(PackerBase):
    """
    VPacker packs its children vertically, automatically adjusting their
    relative positions at draw time.
    """

    def _get_bbox_and_child_offsets(self, renderer):
        dpicor = renderer.points_to_pixels(1.0)
        pad = self.pad * dpicor
        sep = self.sep * dpicor
        if self.width is not None:
            for c in self.get_visible_children():
                if isinstance(c, PackerBase) and c.mode == 'expand':
                    c.set_width(self.width)
        bboxes = [c.get_bbox(renderer) for c in self.get_visible_children()]
        (x0, x1), xoffsets = _get_aligned_offsets([bbox.intervalx for bbox in bboxes], self.width, self.align)
        height, yoffsets = _get_packed_offsets([bbox.height for bbox in bboxes], self.height, sep, self.mode)
        yoffsets = height - (yoffsets + [bbox.y1 for bbox in bboxes])
        ydescent = yoffsets[0]
        yoffsets = yoffsets - ydescent
        return (Bbox.from_bounds(x0, -ydescent, x1 - x0, height).padded(pad), [*zip(xoffsets, yoffsets)])