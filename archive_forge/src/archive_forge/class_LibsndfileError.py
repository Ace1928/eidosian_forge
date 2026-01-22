import os as _os
import sys as _sys
from os import SEEK_SET, SEEK_CUR, SEEK_END
from ctypes.util import find_library as _find_library
from _soundfile import ffi as _ffi
class LibsndfileError(SoundFileRuntimeError):
    """libsndfile errors.


    Attributes
    ----------
    code
        libsndfile internal error number.
    """

    def __init__(self, code, prefix=''):
        SoundFileRuntimeError.__init__(self, code, prefix)
        self.code = code
        self.prefix = prefix

    @property
    def error_string(self):
        """Raw libsndfile error message."""
        if self.code:
            err_str = _snd.sf_error_number(self.code)
            return _ffi.string(err_str).decode('utf-8', 'replace')
        else:
            return '(Garbled error message from libsndfile)'

    def __str__(self):
        return self.prefix + self.error_string