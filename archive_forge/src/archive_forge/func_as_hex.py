import math
import re
from colorsys import hls_to_rgb, rgb_to_hls
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union, cast
from .errors import ColorError
from .utils import Representation, almost_equal_floats
def as_hex(self) -> str:
    """
        Hex string representing the color can be 3, 4, 6 or 8 characters depending on whether the string
        a "short" representation of the color is possible and whether there's an alpha channel.
        """
    values = [float_to_255(c) for c in self._rgba[:3]]
    if self._rgba.alpha is not None:
        values.append(float_to_255(self._rgba.alpha))
    as_hex = ''.join((f'{v:02x}' for v in values))
    if all((c in repeat_colors for c in values)):
        as_hex = ''.join((as_hex[c] for c in range(0, len(as_hex), 2)))
    return '#' + as_hex