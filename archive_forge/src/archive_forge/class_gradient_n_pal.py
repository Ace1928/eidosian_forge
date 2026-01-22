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
class gradient_n_pal(_continuous_color_pal):
    """
    Create a n color gradient palette

    Parameters
    ----------
    colors : list
        list of colors
    values : list, optional
        list of points in the range [0, 1] at which to
        place each color. Must be the same size as
        `colors`. Default to evenly space the colors

    Returns
    -------
    out : function
        Continuous color palette that takes a single
        parameter either a :class:`float` or a sequence
        of floats maps those value(s) onto the palette
        and returns color(s). The float(s) must be
        in the range [0, 1].

    Examples
    --------
    >>> palette = gradient_n_pal(['red', 'blue'])
    >>> palette([0, .25, .5, .75, 1])
    ['#ff0000', '#bf0040', '#7f0080', '#4000bf', '#0000ff']
    >>> palette([-np.inf, 0, np.nan, 1, np.inf])
    [None, '#ff0000', None, '#0000ff', None]
    """
    colors: Sequence[str]
    values: Optional[Sequence[float]] = None

    def __post_init__(self):
        self._cmap = InterpolatedMap(self.colors, self.values)

    def __call__(self, x: FloatArrayLike) -> Sequence[RGBHexColor | None]:
        return self._cmap.continuous_palette(x)