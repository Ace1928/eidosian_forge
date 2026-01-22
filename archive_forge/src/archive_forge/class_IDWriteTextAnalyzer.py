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
class IDWriteTextAnalyzer(com.pIUnknown):
    _methods_ = [('AnalyzeScript', com.STDMETHOD(POINTER(IDWriteTextAnalysisSource), UINT32, UINT32, POINTER(IDWriteTextAnalysisSink))), ('AnalyzeBidi', com.STDMETHOD()), ('AnalyzeNumberSubstitution', com.STDMETHOD()), ('AnalyzeLineBreakpoints', com.STDMETHOD()), ('GetGlyphs', com.STDMETHOD(c_wchar_p, UINT32, IDWriteFontFace, BOOL, BOOL, POINTER(DWRITE_SCRIPT_ANALYSIS), c_wchar_p, c_void_p, POINTER(POINTER(DWRITE_TYPOGRAPHIC_FEATURES)), POINTER(UINT32), UINT32, UINT32, POINTER(UINT16), POINTER(DWRITE_SHAPING_TEXT_PROPERTIES), POINTER(UINT16), POINTER(DWRITE_SHAPING_GLYPH_PROPERTIES), POINTER(UINT32))), ('GetGlyphPlacements', com.STDMETHOD(c_wchar_p, POINTER(UINT16), POINTER(DWRITE_SHAPING_TEXT_PROPERTIES), UINT32, POINTER(UINT16), POINTER(DWRITE_SHAPING_GLYPH_PROPERTIES), UINT32, IDWriteFontFace, FLOAT, BOOL, BOOL, POINTER(DWRITE_SCRIPT_ANALYSIS), c_wchar_p, POINTER(DWRITE_TYPOGRAPHIC_FEATURES), POINTER(UINT32), UINT32, POINTER(FLOAT), POINTER(DWRITE_GLYPH_OFFSET))), ('GetGdiCompatibleGlyphPlacements', com.STDMETHOD())]