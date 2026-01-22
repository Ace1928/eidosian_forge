import struct
import time
import aiokafka.codec as codecs
from aiokafka.errors import CorruptRecordException, UnsupportedCodecError
from aiokafka.util import NO_EXTENSIONS
from aiokafka.codec import (
from .util import decode_varint, encode_varint, calc_crc32c, size_of_varint
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