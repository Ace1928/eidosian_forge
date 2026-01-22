from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
def _init_pointer(self, pointer):
    self._pointer = ffi.gc(pointer, _keepref(cairo, cairo.cairo_font_options_destroy))
    self._check_status()