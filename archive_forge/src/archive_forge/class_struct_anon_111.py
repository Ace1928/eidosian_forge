from OpenGL import platform as _p, constant, extensions
from ctypes import *
from OpenGL.raw.GL._types import *
from OpenGL._bytes import as_8_bit
class struct_anon_111(Structure):
    __slots__ = ['event_type', 'draw_type', 'serial', 'send_event', 'display', 'drawable', 'buffer_mask', 'aux_buffer', 'x', 'y', 'width', 'height', 'count']