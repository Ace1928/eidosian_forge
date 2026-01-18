from __future__ import annotations
import logging # isort:skip
import math
from copy import deepcopy
from typing import TYPE_CHECKING
import numpy as np
from .colors.util import RGB, NamedColor
def viridis(n: int) -> Palette:
    """ Generate a palette of colors from the Viridis palette.

    The full Viridis palette that serves as input for deriving new palettes
    has 256 colors, and looks like:

    :bokeh-palette:`viridis(256)`

    Args:
        n (int) : size of the palette to generate

    Returns:
        seq[str] : a sequence of hex RGB color strings

    Raises:
        ValueError if n is greater than the base palette length of 256

    Examples:

    .. code-block:: python

        >>> viridis(6)
        ('#440154', '#404387', '#29788E', '#22A784', '#79D151', '#FDE724')

    The resulting palette looks like: :bokeh-palette:`viridis(6)`

    """
    return linear_palette(Viridis256, n)