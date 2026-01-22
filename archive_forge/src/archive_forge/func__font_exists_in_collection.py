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
def _font_exists_in_collection(font_collection: ctypes.c_void_p, name: str) -> bool:
    font_name = ctypes.create_unicode_buffer(32)
    for gpfamily in _get_font_families(font_collection):
        gdiplus.GdipGetFamilyName(gpfamily, font_name, '\x00')
        if font_name.value == name:
            return True
    return False