from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, fields
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, cast
from matplotlib._tight_layout import get_subplotspec_list
from ..facets import facet_grid, facet_null, facet_wrap
from .utils import bbox_in_figure_space, tight_bbox_in_figure_space
@dataclass
class WHSpaceParts:
    """
    Width-Height Spaces

    We need these in one places for easy access
    """
    W: float
    H: float
    w: float
    h: float
    sw: float
    sh: float
    wspace: float
    hspace: float