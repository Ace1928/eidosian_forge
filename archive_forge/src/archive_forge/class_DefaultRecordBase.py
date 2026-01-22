import struct
import time
import aiokafka.codec as codecs
from aiokafka.errors import CorruptRecordException, UnsupportedCodecError
from aiokafka.util import NO_EXTENSIONS
from aiokafka.codec import (
from .util import decode_varint, encode_varint, calc_crc32c, size_of_varint
class DefaultRecordBase:
    __slots__ = ()
    HEADER_STRUCT = struct.Struct('>qiibIhiqqqhii')
    ATTRIBUTES_OFFSET = struct.calcsize('>qiibI')
    CRC_OFFSET = struct.calcsize('>qiib')
    AFTER_LEN_OFFSET = struct.calcsize('>qi')
    CODEC_MASK = 7
    CODEC_NONE = 0
    CODEC_GZIP = 1
    CODEC_SNAPPY = 2
    CODEC_LZ4 = 3
    CODEC_ZSTD = 4
    TIMESTAMP_TYPE_MASK = 8
    TRANSACTIONAL_MASK = 16
    CONTROL_MASK = 32
    LOG_APPEND_TIME = 1
    CREATE_TIME = 0
    NO_PARTITION_LEADER_EPOCH = -1

    def _assert_has_codec(self, compression_type):
        if compression_type == self.CODEC_GZIP:
            checker, name = (codecs.has_gzip, 'gzip')
        elif compression_type == self.CODEC_SNAPPY:
            checker, name = (codecs.has_snappy, 'snappy')
        elif compression_type == self.CODEC_LZ4:
            checker, name = (codecs.has_lz4, 'lz4')
        elif compression_type == self.CODEC_ZSTD:
            checker, name = (codecs.has_zstd, 'zstd')
        else:
            raise UnsupportedCodecError(f'Unknown compression codec {compression_type:#04x}')
        if not checker():
            raise UnsupportedCodecError(f'Libraries for {name} compression codec not found')