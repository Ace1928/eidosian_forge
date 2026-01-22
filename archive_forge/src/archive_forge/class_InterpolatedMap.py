from __future__ import annotations
import typing
from dataclasses import dataclass
import numpy as np
from ..hsluv import hex_to_rgb, rgb_to_hex
from ._colormap import ColorMap, ColorMapKind
@dataclass
class InterpolatedMap(_InterpolatedGen):
    colors: Sequence[RGBHexColor] | Sequence[RGBColor] | RGBColorArray
    values: Optional[Sequence[float]] = None
    kind: ColorMapKind = ColorMapKind.miscellaneous

    def __post_init__(self):
        from ..named_colors import get_named_color
        if self.values is None:
            values = np.linspace(0, 1, len(self.colors))
        elif len(self.colors) < 2:
            raise ValueError('A color gradient needs two or more colors')
        else:
            values = np.asarray(self.values)
            if values[0] != 0 or values[-1] != 1:
                raise ValueError(f'Value points of a color gradient should startwith 0 and end with 1. Got {values[0]} and {values[-1]}')
        if len(self.colors) != len(values):
            raise ValueError(f'The values and the colors are different lengthscolors={len(self.colors)}, values={len(values)}')
        if isinstance(self.colors[0], str):
            colors = [hex_to_rgb(get_named_color(c)) for c in self.colors]
        else:
            colors = self.colors
        self._data = np.asarray(colors)
        self._r_lookup = interp_lookup(values, self._data[:, 0])
        self._g_lookup = interp_lookup(values, self._data[:, 1])
        self._b_lookup = interp_lookup(values, self._data[:, 2])