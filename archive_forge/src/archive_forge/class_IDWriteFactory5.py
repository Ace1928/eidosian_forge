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
class IDWriteFactory5(IDWriteFactory4, IDWriteFactory3, IDWriteFactory2, IDWriteFactory1, IDWriteFactory, com.pIUnknown):
    _methods_ = [('CreateFontSetBuilder1', com.STDMETHOD(POINTER(IDWriteFontSetBuilder1))), ('CreateInMemoryFontFileLoader', com.STDMETHOD(POINTER(IDWriteInMemoryFontFileLoader))), ('CreateHttpFontFileLoader', com.STDMETHOD()), ('AnalyzeContainerType', com.STDMETHOD())]