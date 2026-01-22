import os as _os
import sys as _sys
from os import SEEK_SET, SEEK_CUR, SEEK_END
from ctypes.util import find_library as _find_library
from _soundfile import ffi as _ffi
def _init_virtual_io(self, file):
    """Initialize callback functions for sf_open_virtual()."""

    @_ffi.callback('sf_vio_get_filelen')
    def vio_get_filelen(user_data):
        curr = file.tell()
        file.seek(0, SEEK_END)
        size = file.tell()
        file.seek(curr, SEEK_SET)
        return size

    @_ffi.callback('sf_vio_seek')
    def vio_seek(offset, whence, user_data):
        file.seek(offset, whence)
        return file.tell()

    @_ffi.callback('sf_vio_read')
    def vio_read(ptr, count, user_data):
        try:
            buf = _ffi.buffer(ptr, count)
            data_read = file.readinto(buf)
        except AttributeError:
            data = file.read(count)
            data_read = len(data)
            buf = _ffi.buffer(ptr, data_read)
            buf[0:data_read] = data
        return data_read

    @_ffi.callback('sf_vio_write')
    def vio_write(ptr, count, user_data):
        buf = _ffi.buffer(ptr, count)
        data = buf[:]
        written = file.write(data)
        if written is None:
            written = count
        return written

    @_ffi.callback('sf_vio_tell')
    def vio_tell(user_data):
        return file.tell()
    self._virtual_io = {'get_filelen': vio_get_filelen, 'seek': vio_seek, 'read': vio_read, 'write': vio_write, 'tell': vio_tell}
    return _ffi.new('SF_VIRTUAL_IO*', self._virtual_io)