import struct
import time
from binascii import crc32
import aiokafka.codec as codecs
from aiokafka.codec import (
from aiokafka.errors import CorruptRecordException, UnsupportedCodecError
from aiokafka.util import NO_EXTENSIONS
class _LegacyRecordBatchPy(LegacyRecordBase):
    is_control_batch = False
    is_transactional = False
    producer_id = None

    def __init__(self, buffer, magic):
        self._buffer = memoryview(buffer)
        self._magic = magic
        offset, length, crc, magic_, attrs, timestamp = self._read_header(0)
        assert length == len(buffer) - self.LOG_OVERHEAD
        assert magic == magic_
        self._offset = offset
        self._crc = crc
        self._timestamp = timestamp
        self._attributes = attrs
        self._decompressed = False

    @property
    def timestamp_type(self):
        """0 for CreateTime; 1 for LogAppendTime; None if unsupported.

        Value is determined by broker; produced messages should always set to 0
        Requires Kafka >= 0.10 / message version >= 1
        """
        if self._magic == 0:
            return None
        elif self._attributes & self.TIMESTAMP_TYPE_MASK:
            return 1
        else:
            return 0

    @property
    def compression_type(self):
        return self._attributes & self.CODEC_MASK

    @property
    def next_offset(self):
        return self._offset + 1

    def validate_crc(self):
        crc = crc32(self._buffer[self.MAGIC_OFFSET:])
        return self._crc == crc

    def _decompress(self, key_offset):
        pos = key_offset
        key_size = struct.unpack_from('>i', self._buffer, pos)[0]
        pos += self.KEY_LENGTH
        if key_size != -1:
            pos += key_size
        value_size = struct.unpack_from('>i', self._buffer, pos)[0]
        pos += self.VALUE_LENGTH
        if value_size == -1:
            raise CorruptRecordException('Value of compressed message is None')
        else:
            data = self._buffer[pos:pos + value_size]
        compression_type = self.compression_type
        self._assert_has_codec(compression_type)
        if compression_type == self.CODEC_GZIP:
            uncompressed = gzip_decode(data)
        elif compression_type == self.CODEC_SNAPPY:
            uncompressed = snappy_decode(data.tobytes())
        elif compression_type == self.CODEC_LZ4:
            if self._magic == 0:
                raise UnsupportedCodecError('LZ4 is not supported for broker version 0.8/0.9')
            else:
                uncompressed = lz4_decode(data.tobytes())
        return uncompressed

    def _read_header(self, pos):
        if self._magic == 0:
            offset, length, crc, magic_read, attrs = self.HEADER_STRUCT_V0.unpack_from(self._buffer, pos)
            timestamp = None
        else:
            offset, length, crc, magic_read, attrs, timestamp = self.HEADER_STRUCT_V1.unpack_from(self._buffer, pos)
        return (offset, length, crc, magic_read, attrs, timestamp)

    def _read_all_headers(self):
        pos = 0
        msgs = []
        buffer_len = len(self._buffer)
        while pos < buffer_len:
            header = self._read_header(pos)
            msgs.append((header, pos))
            pos += self.LOG_OVERHEAD + header[1]
        return msgs

    def _read_key_value(self, pos):
        key_size = struct.unpack_from('>i', self._buffer, pos)[0]
        pos += self.KEY_LENGTH
        if key_size == -1:
            key = None
        else:
            key = self._buffer[pos:pos + key_size].tobytes()
            pos += key_size
        value_size = struct.unpack_from('>i', self._buffer, pos)[0]
        pos += self.VALUE_LENGTH
        if value_size == -1:
            value = None
        else:
            value = self._buffer[pos:pos + value_size].tobytes()
        return (key, value)

    def __iter__(self):
        if self._magic == 1:
            key_offset = self.KEY_OFFSET_V1
        else:
            key_offset = self.KEY_OFFSET_V0
        timestamp_type = self.timestamp_type
        if self.compression_type:
            if not self._decompressed:
                self._buffer = memoryview(self._decompress(key_offset))
                self._decompressed = True
            headers = self._read_all_headers()
            if self._magic > 0:
                msg_header, _ = headers[-1]
                absolute_base_offset = self._offset - msg_header[0]
            else:
                absolute_base_offset = -1
            for header, msg_pos in headers:
                offset, _, crc, _, attrs, timestamp = header
                assert not attrs & self.CODEC_MASK, 'MessageSet at offset %d appears double-compressed. This should not happen -- check your producers!' % offset
                if timestamp_type == self.LOG_APPEND_TIME:
                    timestamp = self._timestamp
                if absolute_base_offset >= 0:
                    offset += absolute_base_offset
                key, value = self._read_key_value(msg_pos + key_offset)
                yield _LegacyRecordPy(offset, timestamp, timestamp_type, key, value, crc)
        else:
            key, value = self._read_key_value(key_offset)
            yield _LegacyRecordPy(self._offset, self._timestamp, timestamp_type, key, value, self._crc)