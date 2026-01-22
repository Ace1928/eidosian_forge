from __future__ import annotations
import logging # isort:skip
from typing import (
import numpy as np
from ..core.property_mixins import FillProps, HatchProps, LineProps
from ..models.glyphs import MultiLine, MultiPolygons
from ..models.renderers import ContourRenderer, GlyphRenderer
from ..models.sources import ColumnDataSource
from ..palettes import interp_palette
from ..plotting._renderer import _process_sequence_literals
from ..util.dataclasses import dataclass, entries
@dataclass(frozen=True)
class FillData(FillCoords):
    """ Complete geometry data for filled polygons over a whole sequence of contour levels.
    """
    lower_levels: ArrayLike
    upper_levels: ArrayLike

    def asdict(self):
        return dict(entries(self))