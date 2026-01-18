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
def update_positions(self, renderer):
    """Update pixel positions for the annotated point, the text, and the arrow."""
    ox0, oy0 = self._get_xy(renderer, self.xybox, self.boxcoords)
    bbox = self.offsetbox.get_bbox(renderer)
    fw, fh = self._box_alignment
    self.offsetbox.set_offset((ox0 - fw * bbox.width - bbox.x0, oy0 - fh * bbox.height - bbox.y0))
    bbox = self.offsetbox.get_window_extent(renderer)
    self.patch.set_bounds(bbox.bounds)
    mutation_scale = renderer.points_to_pixels(self.get_fontsize())
    self.patch.set_mutation_scale(mutation_scale)
    if self.arrowprops:
        arrow_begin = bbox.p0 + bbox.size * self._arrow_relpos
        arrow_end = self._get_position_xy(renderer)
        self.arrow_patch.set_positions(arrow_begin, arrow_end)
        if 'mutation_scale' in self.arrowprops:
            mutation_scale = renderer.points_to_pixels(self.arrowprops['mutation_scale'])
        self.arrow_patch.set_mutation_scale(mutation_scale)
        patchA = self.arrowprops.get('patchA', self.patch)
        self.arrow_patch.set_patchA(patchA)