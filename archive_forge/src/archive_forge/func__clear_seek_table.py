from array import array
from bisect import bisect_right
from os.path import isfile
from struct import Struct
from warnings import warn
from pyzstd.zstdfile import ZstdDecompressReader, ZstdFile, \
def _clear_seek_table(self):
    self._has_checksum = False
    self._seek_frame_size = 0
    self._file_size = 0
    self._frames_count = 0
    self._full_c_size = 0
    self._full_d_size = 0
    if self._read_mode:
        self._cumulated_c_size = array('q', [0])
        self._cumulated_d_size = array('q', [0])
    else:
        self._frames = array('I')