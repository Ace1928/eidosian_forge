from ctypes import *
from pyglet.gl import *
from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.image.codecs import gif
import pyglet.lib
import pyglet.window
class GdkPixBuf:
    """
    Wrapper around GdkPixBuf object.
    """

    def __init__(self, loader, pixbuf):
        self._loader = loader
        self._pixbuf = pixbuf
        gdk.g_object_ref(pixbuf)

    def __del__(self):
        if self._pixbuf is not None:
            gdk.g_object_unref(self._pixbuf)

    def load_next(self):
        return self._pixbuf is not None

    @property
    def width(self):
        assert self._pixbuf is not None
        return gdkpixbuf.gdk_pixbuf_get_width(self._pixbuf)

    @property
    def height(self):
        assert self._pixbuf is not None
        return gdkpixbuf.gdk_pixbuf_get_height(self._pixbuf)

    @property
    def channels(self):
        assert self._pixbuf is not None
        return gdkpixbuf.gdk_pixbuf_get_n_channels(self._pixbuf)

    @property
    def rowstride(self):
        assert self._pixbuf is not None
        return gdkpixbuf.gdk_pixbuf_get_rowstride(self._pixbuf)

    @property
    def has_alpha(self):
        assert self._pixbuf is not None
        return gdkpixbuf.gdk_pixbuf_get_has_alpha(self._pixbuf) == 1

    def get_pixels(self):
        pixels = gdkpixbuf.gdk_pixbuf_get_pixels(self._pixbuf)
        assert pixels is not None
        buf = (c_ubyte * (self.rowstride * self.height))()
        memmove(buf, pixels, self.rowstride * (self.height - 1) + self.width * self.channels)
        return buf

    def to_image(self):
        if self.width < 1 or self.height < 1 or self.channels < 1 or (self.rowstride < 1):
            return None
        pixels = self.get_pixels()
        if self.channels == 3:
            format = 'RGB'
        else:
            format = 'RGBA'
        return ImageData(self.width, self.height, format, pixels, -self.rowstride)