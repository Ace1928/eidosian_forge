from __future__ import annotations
import logging
import sys
from ._deprecate import deprecate
class _PyAccessF(PyAccess):
    """32 bit float access"""

    def _post_init(self, *args, **kwargs):
        self.pixels = ffi.cast('float **', self.image32)

    def get_pixel(self, x, y):
        return self.pixels[y][x]

    def set_pixel(self, x, y, color):
        try:
            self.pixels[y][x] = color
        except TypeError:
            self.pixels[y][x] = color[0]