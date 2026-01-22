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
def MoveNext(self, hasCurrentFile):
    self.current_index += 1
    if self.current_index != len(self._font_data):
        font_file = IDWriteFontFile()
        self._file_loader.SetCurrentFont(self.current_index, self._font_data[self.current_index])
        key = self.current_index
        if not self.current_index in self._keys:
            buffer = pointer(c_uint32(key))
            ptr = cast(buffer, c_void_p)
            self._keys.append(ptr)
        self.factory.CreateCustomFontFileReference(self._keys[self.current_index], sizeof(buffer), self._file_loader, byref(font_file))
        self._font_files.append(font_file)
        hasCurrentFile[0] = 1
    else:
        hasCurrentFile[0] = 0
    pass