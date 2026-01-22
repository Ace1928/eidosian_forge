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
class MyFontFileStream(com.COMObject):
    _interfaces_ = [IDWriteFontFileStream]

    def __init__(self, data):
        super().__init__()
        self._data = data
        self._size = len(data)
        self._ptrs = []

    def ReadFileFragment(self, fragmentStart, fileOffset, fragmentSize, fragmentContext):
        if fileOffset + fragmentSize > self._size:
            return 2147500037
        fragment = self._data[fileOffset:]
        buffer = (ctypes.c_ubyte * len(fragment)).from_buffer(bytearray(fragment))
        ptr = cast(buffer, c_void_p)
        self._ptrs.append(ptr)
        fragmentStart[0] = ptr
        fragmentContext[0] = None
        return 0

    def ReleaseFileFragment(self, fragmentContext):
        return 0

    def GetFileSize(self, fileSize):
        fileSize[0] = self._size
        return 0

    def GetLastWriteTime(self, lastWriteTime):
        return 2147500033