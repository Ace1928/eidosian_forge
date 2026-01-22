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
class LRTBSpaces:
    """
    Space for components in all directions around the panels
    """
    pack: LayoutPack

    def __post_init__(self):
        self.l = left_spaces(self.pack)
        self.r = right_spaces(self.pack)
        self.t = top_spaces(self.pack)
        self.b = bottom_spaces(self.pack)

    @property
    def left(self):
        """
        Left of the panels in figure space
        """
        return self.l.total

    @property
    def right(self):
        """
        Right of the panels in figure space
        """
        return 1 - self.r.total

    @property
    def top(self):
        """
        Top of the panels in figure space
        """
        return 1 - self.t.total

    @property
    def bottom(self):
        """
        Bottom of the panels in figure space
        """
        return self.b.total