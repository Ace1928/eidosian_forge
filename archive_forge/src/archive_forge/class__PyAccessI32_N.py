from __future__ import annotations
import logging
import sys
from ._deprecate import deprecate
class _PyAccessI32_N(PyAccess):
    """Signed Int32 access, native endian"""

    def _post_init(self, *args, **kwargs):
        self.pixels = self.image32

    def get_pixel(self, x, y):
        return self.pixels[y][x]

    def set_pixel(self, x, y, color):
        self.pixels[y][x] = color