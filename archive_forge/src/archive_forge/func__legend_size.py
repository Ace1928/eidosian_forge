from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, fields
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, cast
from matplotlib._tight_layout import get_subplotspec_list
from ..facets import facet_grid, facet_null, facet_wrap
from .utils import bbox_in_figure_space, tight_bbox_in_figure_space
@cached_property
def _legend_size(self) -> TupleFloat2:
    if not (self.pack.legends and self.pack.legends.bottom):
        return (0, 0)
    bbox = bbox_in_figure_space(self.pack.legends.bottom.box, self.pack.figure, self.pack.renderer)
    return (bbox.width, bbox.height)