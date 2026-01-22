import codecs
from datetime import timezone
from datetime import datetime
from enum import Enum
from functools import total_ordering
from io import BytesIO
import itertools
import logging
import math
import os
import string
import struct
import sys
import time
import types
import warnings
import zlib
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib import _api, _text_helpers, _type1font, cbook, dviread
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.figure import Figure
from matplotlib.font_manager import get_font, fontManager as _fontManager
from matplotlib._afm import AFM
from matplotlib.ft2font import (FIXED_WIDTH, ITALIC, LOAD_NO_SCALE,
from matplotlib.transforms import Affine2D, BboxBase
from matplotlib.path import Path
from matplotlib.dates import UTC
from matplotlib import _path
from . import _backend_pdf_ps
def _draw_xobject_glyph(self, font, fontsize, glyph_idx, x, y):
    """Draw a multibyte character from a Type 3 font as an XObject."""
    glyph_name = font.get_glyph_name(glyph_idx)
    name = self.file._get_xobject_glyph_name(font.fname, glyph_name)
    self.file.output(Op.gsave, 0.001 * fontsize, 0, 0, 0.001 * fontsize, x, y, Op.concat_matrix, Name(name), Op.use_xobject, Op.grestore)