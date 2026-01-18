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
@staticmethod
def version_to_string(version):
    """Return the string representation of the given :ref:`SVG_VERSION`.
        See :meth:`get_versions` for a way to get
        the list of valid version ids.

        """
    c_string = cairo.cairo_svg_version_to_string(version)
    if c_string == ffi.NULL:
        raise ValueError(version)
    return ffi.string(c_string).decode('ascii')