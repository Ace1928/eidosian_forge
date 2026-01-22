import os as _os
import sys as _sys
from os import SEEK_SET, SEEK_CUR, SEEK_END
from ctypes.util import find_library as _find_library
from _soundfile import ffi as _ffi
def _cdata_io(self, action, data, ctype, frames):
    """Call one of libsndfile's read/write functions."""
    assert ctype in _ffi_types.values()
    self._check_if_closed()
    if self.seekable():
        curr = self.tell()
    func = getattr(_snd, 'sf_' + action + 'f_' + ctype)
    frames = func(self._file, data, frames)
    _error_check(self._errorcode)
    if self.seekable():
        self.seek(curr + frames, SEEK_SET)
    return frames