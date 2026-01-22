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
class UniformTextureSequence(TextureSequence):
    """Interface for a sequence of textures, each with the same dimensions.

    :Parameters:
        `item_width` : int
            Width of each texture in the sequence.
        `item_height` : int
            Height of each texture in the sequence.

    """

    def _get_item_width(self):
        raise NotImplementedError('abstract')

    def _get_item_height(self):
        raise NotImplementedError('abstract')

    @property
    def item_width(self):
        return self._get_item_width()

    @property
    def item_height(self):
        return self._get_item_height()