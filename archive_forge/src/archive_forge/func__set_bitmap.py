import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
def _set_bitmap(self, bitmap, close_func=None):
    """Function to set the bitmap and specify the function to unload it."""
    if self._bitmap is not None:
        pass
    if close_func is None:
        close_func = (self._fi.lib.FreeImage_Unload, bitmap)
    self._bitmap = bitmap
    if close_func:
        self._close_funcs.append(close_func)