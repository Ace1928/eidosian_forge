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
Measures the extents of the operations
        stored within the recording-surface.
        This is useful to compute the required size of an image surface
        (or equivalent) into which to replay the full sequence
        of drawing operations.

        :return: A ``(x, y, width, height)`` tuple of floats.

        