from ctypes import *
from pyglet.gl import *
from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.image.codecs import gif
import pyglet.lib
import pyglet.window
def _finish_load(self):
    assert not self.closed
    error = gerror_ptr()
    all_data_passed = gdkpixbuf.gdk_pixbuf_loader_close(self._loader, byref(error))
    self.closed = True
    if not all_data_passed:
        raise ImageDecodeException(_gerror_to_string(error))