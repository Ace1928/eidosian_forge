from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, fields
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, cast
from matplotlib._tight_layout import get_subplotspec_list
from ..facets import facet_grid, facet_null, facet_wrap
from .utils import bbox_in_figure_space, tight_bbox_in_figure_space
def _calculate_panel_spacing_facet_grid(pack: LayoutPack, spaces: LRTBSpaces) -> WHSpaceParts:
    """
    Calculate spacing parts for facet_grid
    """
    pack.facet = cast(facet_grid, pack.facet)
    theme = pack.theme
    ncol = pack.facet.ncol
    nrow = pack.facet.nrow
    W, H = theme.getp('figure_size')
    sw = theme.getp('panel_spacing_x')
    sh = theme.getp('panel_spacing_y') * W / H
    w = (spaces.right - spaces.left - sw * (ncol - 1)) / ncol
    h = (spaces.top - spaces.bottom - sh * (nrow - 1)) / nrow
    wspace = sw / w
    hspace = sh / h
    return WHSpaceParts(W, H, w, h, sw, sh, wspace, hspace)