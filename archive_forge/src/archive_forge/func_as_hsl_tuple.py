import math
import re
from colorsys import hls_to_rgb, rgb_to_hls
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union, cast
from .errors import ColorError
from .utils import Representation, almost_equal_floats
def as_hsl_tuple(self, *, alpha: Optional[bool]=None) -> HslColorTuple:
    """
        Color as an HSL or HSLA tuple, e.g. hue, saturation, lightness and optionally alpha; all elements are in
        the range 0 to 1.

        NOTE: this is HSL as used in HTML and most other places, not HLS as used in python's colorsys.

        :param alpha: whether to include the alpha channel, options are
          None - (default) include alpha only if it's set (e.g. not None)
          True - always include alpha,
          False - always omit alpha,
        """
    h, l, s = rgb_to_hls(self._rgba.r, self._rgba.g, self._rgba.b)
    if alpha is None:
        if self._rgba.alpha is None:
            return (h, s, l)
        else:
            return (h, s, l, self._alpha_float())
    if alpha:
        return (h, s, l, self._alpha_float())
    else:
        return (h, s, l)