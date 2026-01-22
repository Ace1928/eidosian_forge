import struct
import time
from binascii import crc32
import aiokafka.codec as codecs
from aiokafka.codec import (
from aiokafka.errors import CorruptRecordException, UnsupportedCodecError
from aiokafka.util import NO_EXTENSIONS
class _LegacyRecordBatchBuilderPy(LegacyRecordBase):

    def __init__(self, magic, compression_type, batch_size):
        assert magic in [0, 1]
        self._magic = magic
        self._compression_type = compression_type
        self._batch_size = batch_size
        self._msg_buffers = []
        self._pos = 0

    def append(self, offset, timestamp, key, value, headers=None):
        """ Append message to batch.
        """
        if self._magic == 0:
            timestamp = -1
        elif timestamp is None:
            timestamp = int(time.time() * 1000)
        key_size = len(key) if key is not None else 0
        value_size = len(value) if value is not None else 0
        pos = self._pos
        size = self._size_in_bytes(key_size, value_size)
        if offset != 0 and pos + size >= self._batch_size:
            return None
        msg_buf = bytearray(size)
        try:
            crc = self._encode_msg(msg_buf, offset, timestamp, key_size, key, value_size, value)
            self._msg_buffers.append(msg_buf)
            self._pos += size
            return _LegacyRecordMetadataPy(offset, crc, size, timestamp)
        except struct.error:
            if type(offset) != int:
                raise TypeError(offset)
            if type(timestamp) != int:
                raise TypeError(timestamp)
            if not isinstance(key, (bytes, bytearray, memoryview, NoneType)):
                raise TypeError('Unsupported type for key: %s' % type(key))
            if not isinstance(value, (bytes, bytearray, memoryview, NoneType)):
                raise TypeError('Unsupported type for value: %s' % type(value))
            raise

    def _encode_msg(self, buf, offset, timestamp, key_size, key, value_size, value, attributes=0):
        """ Encode msg data into the `msg_buffer`, which should be allocated
            to at least the size of this message.
        """
        magic = self._magic
        length = self.KEY_LENGTH + key_size + self.VALUE_LENGTH + value_size - self.LOG_OVERHEAD
        if magic == 0:
            length += self.KEY_OFFSET_V0
            struct.pack_into('>qiIbbi%dsi%ds' % (key_size, value_size), buf, 0, offset, length, 0, magic, attributes, key_size if key is not None else -1, key or b'', value_size if value is not None else -1, value or b'')
        else:
            length += self.KEY_OFFSET_V1
            struct.pack_into('>qiIbbqi%dsi%ds' % (key_size, value_size), buf, 0, offset, length, 0, magic, attributes, timestamp, key_size if key is not None else -1, key or b'', value_size if value is not None else -1, value or b'')
        crc = crc32(memoryview(buf)[self.MAGIC_OFFSET:])
        struct.pack_into('>I', buf, self.CRC_OFFSET, crc)
        return crc

    def _maybe_compress(self):
        if self._compression_type:
            self._assert_has_codec(self._compression_type)
            buf = self._buffer
            if self._compression_type == self.CODEC_GZIP:
                compressed = gzip_encode(buf)
            elif self._compression_type == self.CODEC_SNAPPY:
                compressed = snappy_encode(buf)
            elif self._compression_type == self.CODEC_LZ4:
                if self._magic == 0:
                    raise UnsupportedCodecError('LZ4 is not supported for broker version 0.8/0.9')
                else:
                    compressed = lz4_encode(bytes(buf))
            compressed_size = len(compressed)
            size = self._size_in_bytes(key_size=0, value_size=compressed_size)
            if size > len(self._buffer):
                self._buffer = bytearray(size)
            else:
                del self._buffer[size:]
            self._encode_msg(self._buffer, offset=0, timestamp=0, key_size=0, key=None, value_size=compressed_size, value=compressed, attributes=self._compression_type)
            self._pos = size
            return True
        return False

    def build(self):
        """Compress batch to be ready for send"""
        self._buffer = bytearray().join(self._msg_buffers)
        self._maybe_compress()
        return self._buffer

    def size(self):
        """ Return current size of data written to buffer
        """
        return self._pos

    def size_in_bytes(self, offset, timestamp, key, value, headers=None):
        """ Actual size of message to add
        """
        assert not headers, 'Headers not supported in v0/v1'
        key_size = len(key) if key is not None else 0
        value_size = len(value) if value is not None else 0
        return self._size_in_bytes(key_size, value_size)

    def _size_in_bytes(self, key_size, value_size):
        return self.LOG_OVERHEAD + self.RECORD_OVERHEAD[self._magic] + key_size + value_size

    @classmethod
    def record_overhead(cls, magic):
        try:
            return cls.RECORD_OVERHEAD[magic]
        except KeyError:
            raise ValueError('Unsupported magic: %d' % magic)