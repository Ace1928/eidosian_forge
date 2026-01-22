import struct
import itertools
from pyglet.gl import *
from pyglet.image import CompressedImageData
from pyglet.image import codecs
from pyglet.image.codecs import s3tc, ImageDecodeException
DDS texture loader.

Reference: http://msdn2.microsoft.com/en-us/library/bb172993.aspx
