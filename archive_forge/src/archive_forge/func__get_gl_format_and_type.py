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
@staticmethod
def _get_gl_format_and_type(fmt):
    if fmt == 'R':
        return (GL_RED, GL_UNSIGNED_BYTE)
    elif fmt == 'RG':
        return (GL_RG, GL_UNSIGNED_BYTE)
    elif fmt == 'RGB':
        return (GL_RGB, GL_UNSIGNED_BYTE)
    elif fmt == 'BGR':
        return (GL_BGR, GL_UNSIGNED_BYTE)
    elif fmt == 'RGBA':
        return (GL_RGBA, GL_UNSIGNED_BYTE)
    elif fmt == 'BGRA':
        return (GL_BGRA, GL_UNSIGNED_BYTE)
    elif fmt == 'L':
        return (GL_LUMINANCE, GL_UNSIGNED_BYTE)
    elif fmt == 'A':
        return (GL_ALPHA, GL_UNSIGNED_BYTE)
    return (None, None)