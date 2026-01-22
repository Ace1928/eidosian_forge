import struct
import time
from binascii import crc32
import aiokafka.codec as codecs
from aiokafka.codec import (
from aiokafka.errors import CorruptRecordException, UnsupportedCodecError
from aiokafka.util import NO_EXTENSIONS
class LegacyRecordBase:
    __slots__ = ()
    HEADER_STRUCT_V0 = struct.Struct('>qiIbb')
    HEADER_STRUCT_V1 = struct.Struct('>qiIbbq')
    LOG_OVERHEAD = CRC_OFFSET = struct.calcsize('>qi')
    MAGIC_OFFSET = LOG_OVERHEAD + struct.calcsize('>I')
    RECORD_OVERHEAD_V0 = struct.calcsize('>Ibbii')
    RECORD_OVERHEAD_V1 = struct.calcsize('>Ibbqii')
    RECORD_OVERHEAD = {0: RECORD_OVERHEAD_V0, 1: RECORD_OVERHEAD_V1}
    KEY_OFFSET_V0 = HEADER_STRUCT_V0.size
    KEY_OFFSET_V1 = HEADER_STRUCT_V1.size
    KEY_LENGTH = VALUE_LENGTH = struct.calcsize('>i')
    CODEC_MASK = 7
    CODEC_GZIP = 1
    CODEC_SNAPPY = 2
    CODEC_LZ4 = 3
    TIMESTAMP_TYPE_MASK = 8
    LOG_APPEND_TIME = 1
    CREATE_TIME = 0

    def _assert_has_codec(self, compression_type):
        if compression_type == self.CODEC_GZIP:
            checker, name = (codecs.has_gzip, 'gzip')
        elif compression_type == self.CODEC_SNAPPY:
            checker, name = (codecs.has_snappy, 'snappy')
        elif compression_type == self.CODEC_LZ4:
            checker, name = (codecs.has_lz4, 'lz4')
        else:
            raise UnsupportedCodecError(f'Unknown compression codec {compression_type:#04x}')
        if not checker():
            raise UnsupportedCodecError(f'Libraries for {name} compression codec not found')