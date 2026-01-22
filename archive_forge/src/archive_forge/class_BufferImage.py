import re
import weakref
from ctypes import *
from io import open, BytesIO
import pyglet
from pyglet.gl import *
from pyglet.gl import gl_info
from pyglet.util import asbytes
from .codecs import ImageEncodeException, ImageDecodeException
from .codecs import registry as _codec_registry
from .codecs import add_default_codecs as _add_default_codecs
from .animation import Animation, AnimationFrame
from .buffer import *
from . import atlas
class BufferImage(AbstractImage):
    """An abstract framebuffer.
    """
    gl_buffer = GL_BACK
    gl_format = 0
    format = ''
    owner = None

    def __init__(self, x, y, width, height):
        super().__init__(width, height)
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def get_image_data(self):
        buf = (GLubyte * (len(self.format) * self.width * self.height))()
        x = self.x
        y = self.y
        if self.owner:
            x += self.owner.x
            y += self.owner.y
        glReadBuffer(self.gl_buffer)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        glReadPixels(x, y, self.width, self.height, self.gl_format, GL_UNSIGNED_BYTE, buf)
        return ImageData(self.width, self.height, self.format, buf)

    def get_region(self, x, y, width, height):
        if self.owner:
            return self.owner.get_region(x + self.x, y + self.y, width, height)
        region = self.__class__(x + self.x, y + self.y, width, height)
        region.gl_buffer = self.gl_buffer
        region.owner = self
        return region