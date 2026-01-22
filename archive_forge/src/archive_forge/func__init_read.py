import struct, sys, time, os
import zlib
import builtins
import io
import _compression
def _init_read(self):
    self._crc = zlib.crc32(b'')
    self._stream_size = 0