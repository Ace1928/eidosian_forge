from contextlib import nullcontext
from math import radians, cos, sin
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib.backend_bases import (
from matplotlib.font_manager import fontManager as _fontManager, get_font
from matplotlib.ft2font import (LOAD_FORCE_AUTOHINT, LOAD_NO_HINTING,
from matplotlib.mathtext import MathTextParser
from matplotlib.path import Path
from matplotlib.transforms import Bbox, BboxBase
from matplotlib.backends._backend_agg import RendererAgg as _RendererAgg
def _prepare_font(self, font_prop):
    """
        Get the `.FT2Font` for *font_prop*, clear its buffer, and set its size.
        """
    font = get_font(_fontManager._find_fonts_by_props(font_prop))
    font.clear()
    size = font_prop.get_size_in_points()
    font.set_size(size, self.dpi)
    return font