import os as _os
import sys as _sys
from os import SEEK_SET, SEEK_CUR, SEEK_END
from ctypes.util import find_library as _find_library
from _soundfile import ffi as _ffi
def _prepare_read(self, start, stop, frames):
    """Seek to start frame and calculate length."""
    if start != 0 and (not self.seekable()):
        raise ValueError('start is only allowed for seekable files')
    if frames >= 0 and stop is not None:
        raise TypeError('Only one of {frames, stop} may be used')
    start, stop, _ = slice(start, stop).indices(self.frames)
    if stop < start:
        stop = start
    if frames < 0:
        frames = stop - start
    if self.seekable():
        self.seek(start, SEEK_SET)
    return frames