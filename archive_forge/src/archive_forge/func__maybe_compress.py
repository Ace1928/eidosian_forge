import struct
import time
import aiokafka.codec as codecs
from aiokafka.errors import CorruptRecordException, UnsupportedCodecError
from aiokafka.util import NO_EXTENSIONS
from aiokafka.codec import (
from .util import decode_varint, encode_varint, calc_crc32c, size_of_varint
def _maybe_compress(self):
    if self._compression_type != self.CODEC_NONE:
        self._assert_has_codec(self._compression_type)
        header_size = self.HEADER_STRUCT.size
        data = bytes(self._buffer[header_size:])
        if self._compression_type == self.CODEC_GZIP:
            compressed = gzip_encode(data)
        elif self._compression_type == self.CODEC_SNAPPY:
            compressed = snappy_encode(data)
        elif self._compression_type == self.CODEC_LZ4:
            compressed = lz4_encode(data)
        elif self._compression_type == self.CODEC_ZSTD:
            compressed = zstd_encode(data)
        compressed_size = len(compressed)
        if len(data) <= compressed_size:
            return False
        else:
            needed_size = header_size + compressed_size
            del self._buffer[needed_size:]
            self._buffer[header_size:needed_size] = compressed
            return True
    return False