from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *

            Uses a metadata name and reader to return a single value. Can be used to get metadata from images.
            If failure, returns 0.
            Also handles cleanup of PROPVARIANT.
        