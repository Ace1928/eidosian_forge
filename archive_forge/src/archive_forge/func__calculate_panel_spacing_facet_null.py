from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, fields
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, cast
from matplotlib._tight_layout import get_subplotspec_list
from ..facets import facet_grid, facet_null, facet_wrap
from .utils import bbox_in_figure_space, tight_bbox_in_figure_space
def _calculate_panel_spacing_facet_null(pack: LayoutPack, spaces: LRTBSpaces) -> WHSpaceParts:
    """
    Calculate spacing parts for facet_null
    """
    W, H = pack.theme.getp('figure_size')
    w = spaces.right - spaces.left
    h = spaces.top - spaces.bottom
    return WHSpaceParts(W, H, w, h, 0, 0, 0, 0)