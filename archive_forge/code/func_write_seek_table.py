from array import array
from bisect import bisect_right
from os.path import isfile
from struct import Struct
from warnings import warn
from pyzstd.zstdfile import ZstdDecompressReader, ZstdFile, \
def write_seek_table(self, fp):
    if self._frames_count > 4294967295:
        warn("SeekableZstdFile's seek table has %d entries, which exceeds the maximal value allowed by Zstandard Seekable Format (0xFFFFFFFF). The entries will be merged into 0xFFFFFFFF entries, this may reduce seeking performance." % self._frames_count, RuntimeWarning, 3)
        self._merge_frames(4294967295)
    offset = 0
    size = 17 + 8 * self._frames_count
    ba = bytearray(size)
    self._s_2uint32.pack_into(ba, offset, 407710302, size - 8)
    offset += 8
    for i in range(0, len(self._frames), 2):
        self._s_2uint32.pack_into(ba, offset, self._frames[i], self._frames[i + 1])
        offset += 8
    self._s_footer.pack_into(ba, offset, self._frames_count, 0, 2408770225)
    fp.write(ba)