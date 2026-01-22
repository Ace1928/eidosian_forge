from ctypes import *
from pyglet.gl import *
from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.image.codecs import gif
import pyglet.lib
import pyglet.window
class GdkPixBufLoader:
    """
    Wrapper around GdkPixBufLoader object.
    """

    def __init__(self, filename, file):
        self.closed = False
        self._file = file
        self._filename = filename
        self._loader = gdkpixbuf.gdk_pixbuf_loader_new()
        if self._loader is None:
            raise ImageDecodeException('Unable to instantiate gdk pixbuf loader')
        self._load_file()

    def __del__(self):
        if self._loader is not None:
            if not self.closed:
                self._cancel_load()
            gdk.g_object_unref(self._loader)

    def _load_file(self):
        self._file.seek(0)
        data = self._file.read()
        self.write(data)

    def _finish_load(self):
        assert not self.closed
        error = gerror_ptr()
        all_data_passed = gdkpixbuf.gdk_pixbuf_loader_close(self._loader, byref(error))
        self.closed = True
        if not all_data_passed:
            raise ImageDecodeException(_gerror_to_string(error))

    def _cancel_load(self):
        assert not self.closed
        gdkpixbuf.gdk_pixbuf_loader_close(self._loader, None)
        self.closed = True

    def write(self, data):
        assert not self.closed, 'Cannot write after closing loader'
        error = gerror_ptr()
        if not gdkpixbuf.gdk_pixbuf_loader_write(self._loader, data, len(data), byref(error)):
            raise ImageDecodeException(_gerror_to_string(error))

    def get_pixbuf(self):
        self._finish_load()
        pixbuf = gdkpixbuf.gdk_pixbuf_loader_get_pixbuf(self._loader)
        if pixbuf is None:
            raise ImageDecodeException('Failed to get pixbuf from loader')
        return GdkPixBuf(self, pixbuf)

    def get_animation(self):
        self._finish_load()
        anim = gdkpixbuf.gdk_pixbuf_loader_get_animation(self._loader)
        if anim is None:
            raise ImageDecodeException('Failed to get animation from loader')
        gif_delays = self._get_gif_delays()
        return GdkPixBufAnimation(self, anim, gif_delays)

    def _get_gif_delays(self):
        assert self._file is not None
        self._file.seek(0)
        gif_stream = gif.read(self._file)
        return [image.delay for image in gif_stream.images]