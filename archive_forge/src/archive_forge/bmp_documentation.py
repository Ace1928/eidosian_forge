import ctypes
from pyglet.image import ImageData
from pyglet.image.codecs import ImageDecoder, ImageDecodeException
Decoder for BMP files.

Currently supports version 3 and 4 bitmaps with BI_RGB and BI_BITFIELDS
encoding.  Alpha channel is supported for 32-bit BI_RGB only.
