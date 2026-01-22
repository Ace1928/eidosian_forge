from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib as mpl
from seaborn._marks.base import (
from typing import TYPE_CHECKING
@document_properties
@dataclass
class Dots(DotBase):
    """
    A dot mark defined by strokes to better handle overplotting.

    See also
    --------
    Dot : A mark suitable for dot plots or less-dense scatterplots.

    Examples
    --------
    .. include:: ../docstrings/objects.Dots.rst

    """
    marker: MappableString = Mappable(rc='scatter.marker', grouping=False)
    pointsize: MappableFloat = Mappable(4, grouping=False)
    stroke: MappableFloat = Mappable(0.75, grouping=False)
    color: MappableColor = Mappable('C0', grouping=False)
    alpha: MappableFloat = Mappable(1, grouping=False)
    fill: MappableBool = Mappable(True, grouping=False)
    fillcolor: MappableColor = Mappable(depend='color', grouping=False)
    fillalpha: MappableFloat = Mappable(0.2, grouping=False)

    def _resolve_properties(self, data, scales):
        resolved = super()._resolve_properties(data, scales)
        resolved['linewidth'] = resolved.pop('stroke')
        resolved['facecolor'] = resolve_color(self, data, 'fill', scales)
        resolved['edgecolor'] = resolve_color(self, data, '', scales)
        resolved.setdefault('edgestyle', (0, None))
        fc = resolved['facecolor']
        if isinstance(fc, tuple):
            resolved['facecolor'] = (fc[0], fc[1], fc[2], fc[3] * resolved['fill'])
        else:
            fc[:, 3] = fc[:, 3] * resolved['fill']
            resolved['facecolor'] = fc
        return resolved