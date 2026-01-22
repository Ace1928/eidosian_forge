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
class IDWriteFontFallback(com.pIUnknown):
    _methods_ = [('MapCharacters', com.STDMETHOD(POINTER(IDWriteTextAnalysisSource), UINT32, UINT32, IDWriteFontCollection, c_wchar_p, DWRITE_FONT_WEIGHT, DWRITE_FONT_STYLE, DWRITE_FONT_STRETCH, POINTER(UINT32), POINTER(IDWriteFont), POINTER(FLOAT)))]