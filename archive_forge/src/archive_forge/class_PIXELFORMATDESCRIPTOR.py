from ctypes import *
from ctypes import _SimpleCData, _check_size
from OpenGL import extensions
from OpenGL.raw.GL._types import *
from OpenGL._bytes import as_8_bit
from OpenGL._opaque import opaque_pointer_cls as _opaque_pointer_cls
class PIXELFORMATDESCRIPTOR(Structure):
    _fields_ = [('nSize', WORD), ('nVersion', WORD), ('dwFlags', DWORD), ('iPixelType', BYTE), ('cColorBits', BYTE), ('cRedBits', BYTE), ('cRedShift', BYTE), ('cGreenBits', BYTE), ('cGreenShift', BYTE), ('cBlueBits', BYTE), ('cBlueShift', BYTE), ('cAlphaBits', BYTE), ('cAlphaShift', BYTE), ('cAccumBits', BYTE), ('cAccumRedBits', BYTE), ('cAccumGreenBits', BYTE), ('cAccumBlueBits', BYTE), ('cAccumAlphaBits', BYTE), ('cAccumDepthBits', BYTE), ('cAccumStencilBits', BYTE), ('cAuxBuffers', BYTE), ('iLayerType', BYTE), ('bReserved', BYTE), ('dwLayerMask', DWORD), ('dwVisibleMask', DWORD), ('dwDamageMask', DWORD)]