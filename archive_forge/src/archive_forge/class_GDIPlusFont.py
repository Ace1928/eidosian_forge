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
class GDIPlusFont(Win32Font):
    glyph_renderer_class = GDIPlusGlyphRenderer
    _private_collection = None
    _system_collection = None
    _default_name = 'Arial'

    def __init__(self, name: str, size: float, bold: bool=False, italic: bool=False, stretch: bool=False, dpi: Optional[float]=None) -> None:
        if not name:
            name = self._default_name
        if stretch:
            warnings.warn('The current font render does not support stretching.')
        super().__init__(name, size, bold, italic, stretch, dpi)
        self._name = name
        family = ctypes.c_void_p()
        if name[0] == '@':
            name = name[1:]
        name = ctypes.c_wchar_p(name)
        if self._private_collection:
            gdiplus.GdipCreateFontFamilyFromName(name, self._private_collection, ctypes.byref(family))
        if not family:
            if _debug_font:
                print(f"Warning: Font '{name}' was not found. Defaulting to: {self._default_name}")
            gdiplus.GdipCreateFontFamilyFromName(name, None, ctypes.byref(family))
        if not family:
            self._name = self._default_name
            gdiplus.GdipCreateFontFamilyFromName(ctypes.c_wchar_p(self._name), None, ctypes.byref(family))
        if dpi is None:
            unit = UnitPoint
            self.dpi = 96
        else:
            unit = UnitPixel
            size = size * dpi // 72
            self.dpi = dpi
        style = 0
        if bold:
            style |= FontStyleBold
        if italic:
            style |= FontStyleItalic
        self._gdipfont = ctypes.c_void_p()
        gdiplus.GdipCreateFont(family, ctypes.c_float(size), style, unit, ctypes.byref(self._gdipfont))
        gdiplus.GdipDeleteFontFamily(family)

    @property
    def name(self) -> str:
        return self._name

    def __del__(self) -> None:
        gdi32.DeleteObject(self.hfont)
        gdiplus.GdipDeleteFont(self._gdipfont)

    @classmethod
    def add_font_data(cls, data: bytes) -> None:
        numfonts = ctypes.c_uint32()
        _handle = gdi32.AddFontMemResourceEx(data, len(data), 0, ctypes.byref(numfonts))
        if _handle is None:
            raise ctypes.WinError()
        if not cls._private_collection:
            cls._private_collection = ctypes.c_void_p()
            gdiplus.GdipNewPrivateFontCollection(ctypes.byref(cls._private_collection))
        gdiplus.GdipPrivateAddMemoryFont(cls._private_collection, data, len(data))

    @classmethod
    def have_font(cls, name: str) -> bool:
        if cls._private_collection:
            if _font_exists_in_collection(cls._private_collection, name):
                return True
        family = ctypes.c_void_p()
        status = gdiplus.GdipCreateFontFamilyFromName(name, None, ctypes.byref(family))
        if status == 0:
            gdiplus.GdipDeleteFontFamily(family)
            return True
        return False