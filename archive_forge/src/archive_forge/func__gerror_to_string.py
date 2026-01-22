from ctypes import *
from pyglet.gl import *
from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.image.codecs import gif
import pyglet.lib
import pyglet.window
def _gerror_to_string(error):
    """
    Convert a GError to a string.
    `error` should be a valid pointer to a GError struct.
    """
    return 'GdkPixBuf Error: domain[{}], code[{}]: {}'.format(error.contents.domain, error.contents.code, error.contents.message)