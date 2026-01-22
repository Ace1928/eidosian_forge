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
class DWRITE_SHAPING_GLYPH_PROPERTIES(ctypes.Structure):
    _fields_ = (('justification', UINT16, 4), ('isClusterStart', UINT16, 1), ('isDiacritic', UINT16, 1), ('isZeroWidthSpace', UINT16, 1), ('reserved', UINT16, 9))