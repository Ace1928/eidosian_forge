from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
def _get_bitmap_frame(self, bitmap_decoder, frame_index):
    bitmap = IWICBitmapFrameDecode()
    bitmap_decoder.GetFrame(frame_index, byref(bitmap))
    return bitmap