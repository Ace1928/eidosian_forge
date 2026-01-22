from __future__ import annotations
import typing
from dataclasses import dataclass
from functools import cached_property
from .._colormaps import PaletteInterpolatedMap
from .._colormaps._colormap import ColorMapKind
@dataclass
class palette:
    name: str
    min_colors: int
    max_colors: int
    swatches: RGB256Swatches
    kind: PaletteKind

    def get_swatch(self, num_colors: int) -> RGB256Swatch:
        """
        Get a swatch with given number of colors
        """
        index = num_colors - self.min_colors
        return self.swatches[index]

    def get_hex_swatch(self, num_colors: int) -> RGBHexSwatch:
        """
        Get a swatch with given number of colors in hex
        """
        swatch = self.get_swatch(num_colors)
        return RGB256Swatch_to_RGBHexSwatch(swatch)

    @cached_property
    def colormap(self) -> PaletteInterpolatedMap:
        """
        Return a colormap representation of the palette
        """
        return PaletteInterpolatedMap(self)