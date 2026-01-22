from __future__ import annotations
import ctypes
import math
import warnings
from typing import Optional, Sequence, TYPE_CHECKING
import pyglet
import pyglet.image
from pyglet.font import base
from pyglet.image.codecs.gdiplus import ImageLockModeRead, BitmapData
from pyglet.image.codecs.gdiplus import PixelFormat32bppARGB, gdiplus, Rect
from pyglet.libs.win32 import _gdi32 as gdi32, _user32 as user32
from pyglet.libs.win32.types import BYTE, ABC, TEXTMETRIC, LOGFONTW
from pyglet.libs.win32.constants import FW_BOLD, FW_NORMAL, ANTIALIASED_QUALITY
from pyglet.libs.win32.context_managers import device_context
def _get_font_families(font_collection: ctypes.c_void_p) -> Sequence[ctypes.c_void_p]:
    num_count = ctypes.c_int()
    gdiplus.GdipGetFontCollectionFamilyCount(font_collection, ctypes.byref(num_count))
    gpfamilies = (ctypes.c_void_p * num_count.value)()
    numFound = ctypes.c_int()
    gdiplus.GdipGetFontCollectionFamilyList(font_collection, num_count, gpfamilies, ctypes.byref(numFound))
    return gpfamilies