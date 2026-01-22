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
def _print_pil(self, filename_or_obj, fmt, pil_kwargs, metadata=None):
    """
        Draw the canvas, then save it using `.image.imsave` (to which
        *pil_kwargs* and *metadata* are forwarded).
        """
    FigureCanvasAgg.draw(self)
    mpl.image.imsave(filename_or_obj, self.buffer_rgba(), format=fmt, origin='upper', dpi=self.figure.dpi, metadata=metadata, pil_kwargs=pil_kwargs)