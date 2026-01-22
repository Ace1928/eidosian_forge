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
class Win32Surface(Surface):
    """Creates a cairo surface that targets the given DC.

    The DC will be queried for its initial clip extents, and this
    will be used as the size of the cairo surface. The resulting
    surface will always be of format CAIRO_FORMAT_RGB24; should
    you need another surface format, you will need to create one
    through cairo_win32_surface_create_with_dib().

    :param hdc :
        The DC to create a surface for,
        as obtained from ``win32gui.CreateDC``.
        **Note**: this unsafely inteprets an integer as a pointer.
        Make sure it actually points to a valid DC!
    :type hdc: int

    *New in cairocffi 0.8*

    """

    def __init__(self, hdc):
        pointer = cairo.cairo_win32_surface_create(ffi.cast('void*', hdc))
        Surface.__init__(self, pointer)