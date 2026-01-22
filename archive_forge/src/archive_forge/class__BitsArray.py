import ctypes
from pyglet.image import ImageData
from pyglet.image.codecs import ImageDecoder, ImageDecodeException
class _BitsArray(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [('data', bits_type * packed_width * height)]