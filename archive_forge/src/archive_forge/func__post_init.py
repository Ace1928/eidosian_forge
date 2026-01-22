from __future__ import annotations
import logging
import sys
from ._deprecate import deprecate
def _post_init(self, *args, **kwargs):
    self.pixels = ffi.cast('float **', self.image32)