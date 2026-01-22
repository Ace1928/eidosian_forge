from collections import OrderedDict
import logging
import urllib.parse
import numpy as np
from matplotlib import _text_helpers, dviread
from matplotlib.font_manager import (
from matplotlib.ft2font import LOAD_NO_HINTING, LOAD_TARGET_LIGHT
from matplotlib.mathtext import MathTextParser
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
def _revalidate_path(self):
    """
        Update the path if necessary.

        The path for the text is initially create with the font size of
        `.FONT_SCALE`, and this path is rescaled to other size when necessary.
        """
    if self._invalid or self._cached_vertices is None:
        tr = Affine2D().scale(self._size / text_to_path.FONT_SCALE).translate(*self._xy)
        self._cached_vertices = tr.transform(self._vertices)
        self._cached_vertices.flags.writeable = False
        self._invalid = False