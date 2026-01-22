from __future__ import annotations
import colorsys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol
from warnings import warn
import numpy as np
from ._colors import (
from .bounds import rescale
from .utils import identity
@dataclass
class brewer_pal(_discrete_pal):
    """
    Utility for making a brewer palette

    Parameters
    ----------
    type : 'sequential' | 'qualitative' | 'diverging'
        Type of palette. Sequential, Qualitative or
        Diverging. The following abbreviations may
        be used, ``seq``, ``qual`` or ``div``.
    palette : int | str
        Which palette to choose from. If is an integer,
        it must be in the range ``[0, m]``, where ``m``
        depends on the number sequential, qualitative or
        diverging palettes. If it is a string, then it
        is the name of the palette.
    direction : int
        The order of colours in the scale. If -1 the order
        of colors is reversed. The default is 1.

    Returns
    -------
    out : function
        A color palette that takes a single
        :class:`int` parameter ``n`` and returns ``n``
        colors. The maximum value of ``n`` varies
        depending on the parameters.

    Examples
    --------
    >>> brewer_pal()(5)
    ['#EFF3FF', '#BDD7E7', '#6BAED6', '#3182BD', '#08519C']
    >>> brewer_pal('qual')(5)
    ['#7FC97F', '#BEAED4', '#FDC086', '#FFFF99', '#386CB0']
    >>> brewer_pal('qual', 2)(5)
    ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#66A61E']
    >>> brewer_pal('seq', 'PuBuGn')(5)
    ['#F6EFF7', '#BDC9E1', '#67A9CF', '#1C9099', '#016C59']

    The available color names for each palette type can be
    obtained using the following code::

        from mizani._colors.brewer import get_palette_names

        print(get_palette_names("sequential"))
        print(get_palette_names("qualitative"))
        print(get_palette_names("diverging"))
    """
    type: ColorScheme | ColorSchemeShort = 'seq'
    palette: int | str = 1
    direction: Literal[1, -1] = 1

    def __post_init__(self):
        from mizani._colors._palettes.brewer import get_brewer_palette
        if self.direction not in (1, -1):
            raise ValueError('direction should be 1 or -1.')
        self.bpal = get_brewer_palette(self.type, self.palette)

    def __call__(self, n: int) -> Sequence[RGBHexColor | None]:
        _n = min(max(n, self.bpal.min_colors), self.bpal.max_colors)
        color_map = self.bpal.get_hex_swatch(_n)
        colors = color_map[:n]
        if n > self.bpal.max_colors:
            msg = f'Warning message:Brewer palette {self.bpal.name} has a maximum of {self.bpal.max_colors} colors Returning the palette you asked for with that many colors'
            warn(msg)
            colors = list(colors) + [None] * (n - self.bpal.max_colors)
        return colors[::self.direction]