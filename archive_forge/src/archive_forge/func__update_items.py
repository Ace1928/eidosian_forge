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
def _update_items(self):
    if not self._items:
        self._items = []
        y = 0
        for row in range(self.rows):
            x = 0
            for col in range(self.columns):
                self._items.append(self.image.get_region(x, y, self.item_width, self.item_height))
                x += self.item_width + self.column_padding
            y += self.item_height + self.row_padding