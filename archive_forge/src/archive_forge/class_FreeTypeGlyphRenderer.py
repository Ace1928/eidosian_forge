import ctypes
import warnings
from collections import namedtuple
from pyglet.util import asbytes, asstr
from pyglet.font import base
from pyglet import image
from pyglet.font.fontconfig import get_fontconfig
from pyglet.font.freetype_lib import *
class FreeTypeGlyphRenderer(base.GlyphRenderer):

    def __init__(self, font):
        super().__init__(font)
        self.font = font
        self._glyph_slot = None
        self._bitmap = None
        self._width = None
        self._height = None
        self._mode = None
        self._pitch = None
        self._baseline = None
        self._lsb = None
        self._advance_x = None
        self._data = None

    def _get_glyph(self, character):
        assert self.font
        assert len(character) == 1
        self._glyph_slot = self.font.get_glyph_slot(character)
        self._bitmap = self._glyph_slot.bitmap

    def _get_glyph_metrics(self):
        self._width = self._glyph_slot.bitmap.width
        self._height = self._glyph_slot.bitmap.rows
        self._mode = self._glyph_slot.bitmap.pixel_mode
        self._pitch = self._glyph_slot.bitmap.pitch
        self._baseline = self._height - self._glyph_slot.bitmap_top
        self._lsb = self._glyph_slot.bitmap_left
        self._advance_x = int(f26p6_to_float(self._glyph_slot.advance.x))

    def _get_bitmap_data(self):
        if self._mode == FT_PIXEL_MODE_MONO:
            self._convert_mono_to_gray_bitmap()
        elif self._mode == FT_PIXEL_MODE_GRAY:
            assert self._glyph_slot.bitmap.num_grays == 256
            self._data = self._glyph_slot.bitmap.buffer
        else:
            raise base.FontException('Unsupported render mode for this glyph')

    def _convert_mono_to_gray_bitmap(self):
        bitmap_data = cast(self._bitmap.buffer, POINTER(c_ubyte * (self._pitch * self._height))).contents
        data = (c_ubyte * (self._pitch * 8 * self._height))()
        data_i = 0
        for byte in bitmap_data:
            data[data_i + 0] = byte & 128 and 255 or 0
            data[data_i + 1] = byte & 64 and 255 or 0
            data[data_i + 2] = byte & 32 and 255 or 0
            data[data_i + 3] = byte & 16 and 255 or 0
            data[data_i + 4] = byte & 8 and 255 or 0
            data[data_i + 5] = byte & 4 and 255 or 0
            data[data_i + 6] = byte & 2 and 255 or 0
            data[data_i + 7] = byte & 1 and 255 or 0
            data_i += 8
        self._data = data
        self._pitch <<= 3

    def _create_glyph(self):
        img = image.ImageData(self._width, self._height, 'A', self._data, abs(self._pitch))
        if pyglet.gl.current_context.get_info().get_opengl_api() == 'gles':
            GL_ALPHA = 6406
            glyph = self.font.create_glyph(img, fmt=GL_ALPHA)
        else:
            glyph = self.font.create_glyph(img)
        glyph.set_bearings(self._baseline, self._lsb, self._advance_x)
        if self._pitch > 0:
            t = list(glyph.tex_coords)
            glyph.tex_coords = t[9:12] + t[6:9] + t[3:6] + t[:3]
        return glyph

    def render(self, text):
        self._get_glyph(text[0])
        self._get_glyph_metrics()
        self._get_bitmap_data()
        return self._create_glyph()