import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class TEXTMETRIC(Structure):
    _fields_ = [('tmHeight', c_long), ('tmAscent', c_long), ('tmDescent', c_long), ('tmInternalLeading', c_long), ('tmExternalLeading', c_long), ('tmAveCharWidth', c_long), ('tmMaxCharWidth', c_long), ('tmWeight', c_long), ('tmOverhang', c_long), ('tmDigitizedAspectX', c_long), ('tmDigitizedAspectY', c_long), ('tmFirstChar', c_char), ('tmLastChar', c_char), ('tmDefaultChar', c_char), ('tmBreakChar', c_char), ('tmItalic', c_byte), ('tmUnderlined', c_byte), ('tmStruckOut', c_byte), ('tmPitchAndFamily', c_byte), ('tmCharSet', c_byte)]
    __slots__ = [f[0] for f in _fields_]