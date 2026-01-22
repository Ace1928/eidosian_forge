import ctypes
import warnings
from collections import namedtuple
from pyglet.util import asbytes, asstr
from pyglet.font import base
from pyglet import image
from pyglet.font.fontconfig import get_fontconfig
from pyglet.font.freetype_lib import *
def _get_bitmap_data(self):
    if self._mode == FT_PIXEL_MODE_MONO:
        self._convert_mono_to_gray_bitmap()
    elif self._mode == FT_PIXEL_MODE_GRAY:
        assert self._glyph_slot.bitmap.num_grays == 256
        self._data = self._glyph_slot.bitmap.buffer
    else:
        raise base.FontException('Unsupported render mode for this glyph')