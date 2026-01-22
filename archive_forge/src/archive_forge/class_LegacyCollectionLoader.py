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
class LegacyCollectionLoader(com.COMObject):
    _interfaces_ = [IDWriteFontCollectionLoader]

    def __init__(self, factory, loader):
        super().__init__()
        self._enumerator = MyEnumerator(factory, loader)

    def AddFontData(self, fonts):
        self._enumerator.AddFontData(fonts)

    def CreateEnumeratorFromKey(self, factory, key, key_size, enumerator):
        self._ptr = ctypes.cast(self._enumerator.as_interface(IDWriteFontFileEnumerator), POINTER(IDWriteFontFileEnumerator))
        enumerator[0] = self._ptr
        return 0