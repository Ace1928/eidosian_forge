import os as _os
import sys as _sys
from os import SEEK_SET, SEEK_CUR, SEEK_END
from ctypes.util import find_library as _find_library
from _soundfile import ffi as _ffi
def _available_formats_helper(count_flag, format_flag):
    """Helper for available_formats() and available_subtypes()."""
    count = _ffi.new('int*')
    _snd.sf_command(_ffi.NULL, count_flag, count, _ffi.sizeof('int'))
    for format_int in range(count[0]):
        yield _format_info(format_int, format_flag)