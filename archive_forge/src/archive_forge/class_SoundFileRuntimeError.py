import os as _os
import sys as _sys
from os import SEEK_SET, SEEK_CUR, SEEK_END
from ctypes.util import find_library as _find_library
from _soundfile import ffi as _ffi
class SoundFileRuntimeError(SoundFileError, RuntimeError):
    """soundfile module runtime error.

    Errors that used to be `RuntimeError`."""
    pass