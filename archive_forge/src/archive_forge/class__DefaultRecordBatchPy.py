import struct
import time
import aiokafka.codec as codecs
from aiokafka.errors import CorruptRecordException, UnsupportedCodecError
from aiokafka.util import NO_EXTENSIONS
from aiokafka.codec import (
from .util import decode_varint, encode_varint, calc_crc32c, size_of_varint
class _DefaultRecordBatchPy(DefaultRecordBase):

    def __init__(self, buffer):
        self._buffer = bytearray(buffer)
        self._header_data = self.HEADER_STRUCT.unpack_from(self._buffer)
        self._pos = self.HEADER_STRUCT.size
        self._num_records = self._header_data[12]
        self._next_record_index = 0
        self._decompressed = False

    @property
    def base_offset(self):
        return self._header_data[0]

    @property
    def magic(self):
        return self._header_data[3]

    @property
    def crc(self):
        return self._header_data[4]

    @property
    def attributes(self):
        return self._header_data[5]

    @property
    def compression_type(self):
        return self.attributes & self.CODEC_MASK

    @property
    def timestamp_type(self):
        return int(bool(self.attributes & self.TIMESTAMP_TYPE_MASK))

    @property
    def is_transactional(self):
        return bool(self.attributes & self.TRANSACTIONAL_MASK)

    @property
    def is_control_batch(self):
        return bool(self.attributes & self.CONTROL_MASK)

    @property
    def last_offset_delta(self):
        return self._header_data[6]

    @property
    def first_timestamp(self):
        return self._header_data[7]

    @property
    def max_timestamp(self):
        return self._header_data[8]

    @property
    def producer_id(self):
        return self._header_data[9]

    @property
    def producer_epoch(self):
        return self._header_data[10]

    @property
    def base_sequence(self):
        return self._header_data[11]

    @property
    def next_offset(self):
        return self.base_offset + self.last_offset_delta + 1

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

    def _read_msg(self, decode_varint=decode_varint):
        buffer = self._buffer
        pos = self._pos
        length, pos = decode_varint(buffer, pos)
        start_pos = pos
        _, pos = decode_varint(buffer, pos)
        ts_delta, pos = decode_varint(buffer, pos)
        if self.timestamp_type == self.LOG_APPEND_TIME:
            timestamp = self.max_timestamp
        else:
            timestamp = self.first_timestamp + ts_delta
        offset_delta, pos = decode_varint(buffer, pos)
        offset = self.base_offset + offset_delta
        key_len, pos = decode_varint(buffer, pos)
        if key_len >= 0:
            key = bytes(buffer[pos:pos + key_len])
            pos += key_len
        else:
            key = None
        value_len, pos = decode_varint(buffer, pos)
        if value_len >= 0:
            value = bytes(buffer[pos:pos + value_len])
            pos += value_len
        else:
            value = None
        header_count, pos = decode_varint(buffer, pos)
        if header_count < 0:
            raise CorruptRecordException(f'Found invalid number of record headers {header_count}')
        headers = []
        while header_count:
            h_key_len, pos = decode_varint(buffer, pos)
            if h_key_len < 0:
                raise CorruptRecordException(f'Invalid negative header key size {h_key_len}')
            h_key = buffer[pos:pos + h_key_len].decode('utf-8')
            pos += h_key_len
            h_value_len, pos = decode_varint(buffer, pos)
            if h_value_len >= 0:
                h_value = bytes(buffer[pos:pos + h_value_len])
                pos += h_value_len
            else:
                h_value = None
            headers.append((h_key, h_value))
            header_count -= 1
        if pos - start_pos != length:
            raise CorruptRecordException(f'Invalid record size: expected to read {length} bytes in record payload, but instead read {pos - start_pos}')
        self._pos = pos
        return DefaultRecord(offset, timestamp, self.timestamp_type, key, value, headers)

    def __iter__(self):
        self._maybe_uncompress()
        return self

    def __next__(self):
        if self._next_record_index >= self._num_records:
            if self._pos != len(self._buffer):
                raise CorruptRecordException(f'{len(self._buffer) - self._pos} unconsumed bytes after all records consumed')
            raise StopIteration
        try:
            msg = self._read_msg()
        except (ValueError, IndexError) as err:
            raise CorruptRecordException(f'Found invalid record structure: {err!r}')
        else:
            self._next_record_index += 1
        return msg
    next = __next__

    def validate_crc(self):
        assert self._decompressed is False, 'Validate should be called before iteration'
        crc = self.crc
        data_view = memoryview(self._buffer)[self.ATTRIBUTES_OFFSET:]
        verify_crc = calc_crc32c(data_view.tobytes())
        return crc == verify_crc