import math
import re
from colorsys import hls_to_rgb, rgb_to_hls
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union, cast
from .errors import ColorError
from .utils import Representation, almost_equal_floats
def as_hsl(self) -> str:
    """
        Color as an hsl(<h>, <s>, <l>) or hsl(<h>, <s>, <l>, <a>) string.
        """
    if self._rgba.alpha is None:
        h, s, li = self.as_hsl_tuple(alpha=False)
        return f'hsl({h * 360:0.0f}, {s:0.0%}, {li:0.0%})'
    else:
        h, s, li, a = self.as_hsl_tuple(alpha=True)
        return f'hsl({h * 360:0.0f}, {s:0.0%}, {li:0.0%}, {round(a, 2)})'