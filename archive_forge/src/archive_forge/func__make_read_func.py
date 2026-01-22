import ctypes
import io
import operator
import os
import sys
import weakref
from functools import reduce
from pathlib import Path
from tempfile import NamedTemporaryFile
from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontOptions, _encode_string
def _make_read_func(file_obj):
    """Return a CFFI callback that reads from a file-like object."""

    @ffi.callback('cairo_read_func_t', error=constants.STATUS_READ_ERROR)
    def read_func(_closure, data, length):
        string = file_obj.read(length)
        if len(string) < length:
            return constants.STATUS_READ_ERROR
        ffi.buffer(data, length)[:len(string)] = string
        return constants.STATUS_SUCCESS
    return read_func