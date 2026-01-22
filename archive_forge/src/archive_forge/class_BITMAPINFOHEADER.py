import ctypes
from pyglet.image import ImageData
from pyglet.image.codecs import ImageDecoder, ImageDecodeException
class BITMAPINFOHEADER(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [('biSize', DWORD), ('biWidth', LONG), ('biHeight', LONG), ('biPlanes', WORD), ('biBitCount', WORD), ('biCompression', DWORD), ('biSizeImage', DWORD), ('biXPelsPerMeter', LONG), ('biYPelsPerMeter', LONG), ('biClrUsed', DWORD), ('biClrImportant', DWORD)]