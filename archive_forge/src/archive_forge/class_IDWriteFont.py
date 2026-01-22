import copy
import os
import pathlib
import platform
from ctypes import *
from typing import List, Optional, Tuple
import math
import pyglet
from pyglet.font import base
from pyglet.image.codecs.wic import IWICBitmap, WICDecoder, GUID_WICPixelFormat32bppPBGRA
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
class IDWriteFont(com.pIUnknown):
    _methods_ = [('GetFontFamily', com.STDMETHOD(POINTER(IDWriteFontFamily))), ('GetWeight', com.METHOD(DWRITE_FONT_WEIGHT)), ('GetStretch', com.METHOD(DWRITE_FONT_STRETCH)), ('GetStyle', com.METHOD(DWRITE_FONT_STYLE)), ('IsSymbolFont', com.METHOD(BOOL)), ('GetFaceNames', com.STDMETHOD(POINTER(IDWriteLocalizedStrings))), ('GetInformationalStrings', com.STDMETHOD(DWRITE_INFORMATIONAL_STRING_ID, POINTER(IDWriteLocalizedStrings), POINTER(BOOL))), ('GetSimulations', com.STDMETHOD()), ('GetMetrics', com.STDMETHOD()), ('HasCharacter', com.STDMETHOD(UINT32, POINTER(BOOL))), ('CreateFontFace', com.STDMETHOD(POINTER(IDWriteFontFace)))]