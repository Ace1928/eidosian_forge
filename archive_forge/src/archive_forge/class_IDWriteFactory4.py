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
class IDWriteFactory4(IDWriteFactory3, com.pIUnknown):
    _methods_ = [('TranslateColorGlyphRun4', com.STDMETHOD(D2D_POINT_2F, POINTER(DWRITE_GLYPH_RUN), c_void_p, DWRITE_GLYPH_IMAGE_FORMATS, DWRITE_MEASURING_MODE, c_void_p, UINT32, POINTER(IDWriteColorGlyphRunEnumerator1))), ('ComputeGlyphOrigins_', com.STDMETHOD()), ('ComputeGlyphOrigins', com.STDMETHOD())]