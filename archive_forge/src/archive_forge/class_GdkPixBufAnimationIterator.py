from ctypes import *
from pyglet.gl import *
from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.image.codecs import gif
import pyglet.lib
import pyglet.window
class GdkPixBufAnimationIterator:

    def __init__(self, loader, anim_iter, start_time, gif_delays):
        self._iter = anim_iter
        self._first = True
        self._time = start_time
        self._loader = loader
        self._gif_delays = gif_delays
        self.delay_time = None

    def __del__(self):
        if self._iter is not None:
            gdk.g_object_unref(self._iter)

    def __iter__(self):
        return self

    def __next__(self):
        self._advance()
        frame = self.get_frame()
        if frame is None:
            raise StopIteration
        return frame

    def _advance(self):
        if not self._gif_delays:
            raise StopIteration
        self.delay_time = self._gif_delays.pop(0)
        if self._first:
            self._first = False
        elif self.gdk_delay_time == -1:
            raise StopIteration
        else:
            gdk_delay = self.gdk_delay_time * 1000
            us = self._time.tv_usec + gdk_delay
            self._time.tv_sec += us // 1000000
            self._time.tv_usec = us % 1000000
            gdkpixbuf.gdk_pixbuf_animation_iter_advance(self._iter, byref(self._time))

    def get_frame(self):
        pixbuf = gdkpixbuf.gdk_pixbuf_animation_iter_get_pixbuf(self._iter)
        if pixbuf is None:
            return None
        image = GdkPixBuf(self._loader, pixbuf).to_image()
        return AnimationFrame(image, self.delay_time)

    @property
    def gdk_delay_time(self):
        assert self._iter is not None
        return gdkpixbuf.gdk_pixbuf_animation_iter_get_delay_time(self._iter)