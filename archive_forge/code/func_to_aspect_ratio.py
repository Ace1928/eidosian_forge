from __future__ import annotations
import typing
from copy import deepcopy
from dataclasses import dataclass
from ._plot_side_space import LRTBSpaces, WHSpaceParts, calculate_panel_spacing
from .utils import bbox_in_figure_space, get_transPanels
def to_aspect_ratio(self, facet: facet, ratio: float, parts: WHSpaceParts) -> TightParams:
    """
        Modify TightParams to get a given aspect ratio
        """
    current_ratio = parts.h * parts.H / (parts.w * parts.W)
    increase_aspect_ratio = ratio > current_ratio
    if increase_aspect_ratio:
        return self._reduce_width(facet, ratio, parts)
    else:
        return self._reduce_height(facet, ratio, parts)