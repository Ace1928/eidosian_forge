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
def _filled_to_coords(filled: FillReturn_OuterOffset) -> SingleFillCoords:
    xs = []
    ys = []
    for points, offsets in zip(*filled):
        n = len(offsets) - 1
        xs.append([points[offsets[i]:offsets[i + 1], 0] for i in range(n)])
        ys.append([points[offsets[i]:offsets[i + 1], 1] for i in range(n)])
    return SingleFillCoords(xs, ys)