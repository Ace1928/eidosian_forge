import sys
from array import array
from functools import partial
from io import BytesIO
from . import Context, ImageSurface, constants, dlopen
from .ffi import ffi_pixbuf as ffi
class Pixbuf(object):
    """Wrap a ``GdkPixbuf`` pointer and simulate methods."""

    def __init__(self, pointer):
        gobject.g_object_ref(pointer)
        self._pointer = ffi.gc(pointer, gobject.g_object_unref)

    def __getattr__(self, name):
        function = getattr(gdk_pixbuf, 'gdk_pixbuf_' + name)
        return partial(function, self._pointer)