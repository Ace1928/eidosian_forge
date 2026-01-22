from __future__ import annotations
import logging
import sys
from ._deprecate import deprecate
class _PyAccess32_2(PyAccess):
    """PA, LA, stored in first and last bytes of a 32 bit word"""

    def _post_init(self, *args, **kwargs):
        self.pixels = ffi.cast('struct Pixel_RGBA **', self.image32)

    def get_pixel(self, x, y):
        pixel = self.pixels[y][x]
        return (pixel.r, pixel.a)

    def set_pixel(self, x, y, color):
        pixel = self.pixels[y][x]
        pixel.r = min(color[0], 255)
        pixel.a = min(color[1], 255)