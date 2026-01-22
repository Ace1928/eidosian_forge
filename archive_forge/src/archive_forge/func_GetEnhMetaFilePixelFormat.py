from OpenGL import platform as _p, arrays
from OpenGL.raw.WGL import _types as _cs
from OpenGL.raw.WGL._types import *
from OpenGL.raw.WGL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(_cs.UINT, _cs.HENHMETAFILE, ctypes.POINTER(_cs.PIXELFORMATDESCRIPTOR))
def GetEnhMetaFilePixelFormat(hemf, ppfd):
    pass