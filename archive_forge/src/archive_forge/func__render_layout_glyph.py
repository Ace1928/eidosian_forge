import copy
import os
import pathlib
import platform
from ctypes import *
from typing import List, Optional, Tuple
import math
import pyglet
from pyglet.font import base
from pyglet.image.codecs.wic import IWICBitmap, WICDecoder, GUID_WICPixelFormat32bppPBGRA
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
def _render_layout_glyph(self, text_buffer, i, clusters, check_color=True):
    text_length = clusters.count(i)
    text_index = clusters.index(i)
    actual_text = text_buffer[text_index:text_index + text_length]
    if actual_text not in self.glyphs:
        glyph = self._glyph_renderer.render_using_layout(text_buffer[text_index:text_index + text_length])
        if glyph:
            if check_color and self._glyph_renderer.draw_options & D2D1_DRAW_TEXT_OPTIONS_ENABLE_COLOR_FONT:
                fb_ff = self._get_fallback_font_face(text_index, text_length)
                if fb_ff:
                    glyph.colored = self.is_fallback_str_colored(fb_ff, actual_text)
        else:
            glyph = self._empty_glyph
        self.glyphs[actual_text] = glyph
    return self.glyphs[actual_text]