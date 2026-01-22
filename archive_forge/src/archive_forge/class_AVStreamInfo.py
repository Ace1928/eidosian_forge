from ctypes import c_int, c_int64
from ctypes import c_uint8, c_uint, c_double, c_ubyte, c_size_t, c_char, c_char_p
from ctypes import c_void_p, POINTER, CFUNCTYPE, Structure
import pyglet.lib
from pyglet.util import debug_print
from . import compat
from . import libavcodec
from . import libavutil
class AVStreamInfo(Structure):
    _fields_ = [('last_dts', c_int64), ('duration_gcd', c_int64), ('duration_count', c_int), ('rfps_duration_sum', c_int64), ('duration_error', POINTER(c_double * 2 * (30 * 12 + 30 + 3 + 6))), ('codec_info_duration', c_int64), ('codec_info_duration_fields', c_int64), ('frame_delay_evidence', c_int), ('found_decoder', c_int), ('last_duration', c_int64), ('fps_first_dts', c_int64), ('fps_first_dts_idx', c_int), ('fps_last_dts', c_int64), ('fps_last_dts_idx', c_int)]