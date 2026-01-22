import struct
import time
import aiokafka.codec as codecs
from aiokafka.errors import CorruptRecordException, UnsupportedCodecError
from aiokafka.util import NO_EXTENSIONS
from aiokafka.codec import (
from .util import decode_varint, encode_varint, calc_crc32c, size_of_varint
def _maybe_uncompress(self):
    if not self._decompressed:
        compression_type = self.compression_type
        if compression_type != self.CODEC_NONE:
            self._assert_has_codec(compression_type)
            data = memoryview(self._buffer)[self._pos:]
            if compression_type == self.CODEC_GZIP:
                uncompressed = gzip_decode(data)
            elif compression_type == self.CODEC_SNAPPY:
                uncompressed = snappy_decode(data.tobytes())
            elif compression_type == self.CODEC_LZ4:
                uncompressed = lz4_decode(data.tobytes())
            if compression_type == self.CODEC_ZSTD:
                uncompressed = zstd_decode(data.tobytes())
            self._buffer = bytearray(uncompressed)
            self._pos = 0
    self._decompressed = True