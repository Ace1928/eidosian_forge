import ctypes
import warnings
from collections import namedtuple
from pyglet.util import asbytes, asstr
from pyglet.font import base
from pyglet import image
from pyglet.font.fontconfig import get_fontconfig
from pyglet.font.freetype_lib import *
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