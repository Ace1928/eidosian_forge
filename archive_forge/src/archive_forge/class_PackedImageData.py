import re
import ctypes
from pyglet.gl import *
from pyglet.gl import gl_info
from pyglet.image import AbstractImage, Texture
class PackedImageData(AbstractImage):
    _current_texture = None

    def __init__(self, width, height, fmt, packed_format, data):
        super().__init__(width, height)
        self.format = fmt
        self.packed_format = packed_format
        self.data = data

    def unpack(self):
        if self.packed_format == GL_UNSIGNED_SHORT_5_6_5:
            i = 0
            out = (ctypes.c_ubyte * (self.width * self.height * 3))()
            for c in self.data:
                out[i + 2] = (c & 31) << 3
                out[i + 1] = (c & 2016) >> 3
                out[i] = (c & 63488) >> 8
                i += 3
            self.data = out
            self.packed_format = GL_UNSIGNED_BYTE

    def _get_texture(self):
        if self._current_texture:
            return self._current_texture
        texture = Texture.create(self.width, self.height, GL_TEXTURE_2D, None)
        glBindTexture(texture.target, texture.id)
        glTexParameteri(texture.target, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        if not gl_info.have_version(1, 2) or True:
            self.unpack()
        glTexImage2D(texture.target, texture.level, self.format, self.width, self.height, 0, self.format, self.packed_format, self.data)
        self._current_texture = texture
        return texture
    texture = property(_get_texture)

    def get_texture(self, rectangle=False, force_rectangle=False):
        """The parameters 'rectangle' and 'force_rectangle' are ignored.
           See the documentation of the method 'AbstractImage.get_texture' for
           a more detailed documentation of the method. """
        return self._get_texture()