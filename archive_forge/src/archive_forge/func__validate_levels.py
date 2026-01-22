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
def _validate_levels(levels: ArrayLike | None) -> NDArray[float]:
    levels = np.asarray(levels)
    if levels.ndim == 0 or len(levels) == 0:
        raise ValueError('No contour levels specified')
    if len(levels) > 1 and np.diff(levels).min() <= 0.0:
        raise ValueError('Contour levels must be increasing')
    return levels