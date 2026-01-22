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
class SolidColorImagePattern(ImagePattern):
    """Creates an image filled with a solid color."""

    def __init__(self, color=(0, 0, 0, 0)):
        """Create a solid image pattern with the given color.

        :Parameters:
            `color` : (int, int, int, int)
                4-tuple of ints in range [0,255] giving RGBA components of
                color to fill with.

        """
        self.color = _color_as_bytes(color)

    def create_image(self, width, height):
        data = self.color * width * height
        return ImageData(width, height, 'RGBA', data)