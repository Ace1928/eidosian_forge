import math
import re
from colorsys import hls_to_rgb, rgb_to_hls
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union, cast
from .errors import ColorError
from .utils import Representation, almost_equal_floats
def as_named(self, *, fallback: bool=False) -> str:
    if self._rgba.alpha is None:
        rgb = cast(Tuple[int, int, int], self.as_rgb_tuple())
        try:
            return COLORS_BY_VALUE[rgb]
        except KeyError as e:
            if fallback:
                return self.as_hex()
            else:
                raise ValueError('no named color found, use fallback=True, as_hex() or as_rgb()') from e
    else:
        return self.as_hex()