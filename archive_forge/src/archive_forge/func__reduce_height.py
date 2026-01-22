from __future__ import annotations
import typing
from copy import deepcopy
from dataclasses import dataclass
from ._plot_side_space import LRTBSpaces, WHSpaceParts, calculate_panel_spacing
from .utils import bbox_in_figure_space, get_transPanels
def _reduce_height(self, facet: facet, ratio: float, parts: WHSpaceParts) -> TightParams:
    """
        Reduce the height of axes to get the aspect ratio
        """
    self = deepcopy(self)
    h1 = ratio * parts.w * (parts.W / parts.H)
    dh = (parts.h - h1) * facet.nrow / 2
    self.params.top -= dh
    self.params.bottom += dh
    self.params.hspace = parts.sh / h1
    self.sides.t.plot_margin += dh
    self.sides.b.plot_margin += dh
    return self