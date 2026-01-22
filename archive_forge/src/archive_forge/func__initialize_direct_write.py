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
@classmethod
def _initialize_direct_write(cls):
    """ All direct write fonts needs factory access as well as the loaders."""
    if WINDOWS_10_CREATORS_UPDATE_OR_GREATER:
        cls._write_factory = IDWriteFactory5()
        guid = IID_IDWriteFactory5
    elif WINDOWS_8_1_OR_GREATER:
        cls._write_factory = IDWriteFactory2()
        guid = IID_IDWriteFactory2
    else:
        cls._write_factory = IDWriteFactory()
        guid = IID_IDWriteFactory
    DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, guid, byref(cls._write_factory))