import struct
import time
from binascii import crc32
import aiokafka.codec as codecs
from aiokafka.codec import (
from aiokafka.errors import CorruptRecordException, UnsupportedCodecError
from aiokafka.util import NO_EXTENSIONS
class _LegacyRecordPy:
    __slots__ = ('_offset', '_timestamp', '_timestamp_type', '_key', '_value', '_crc')

    def __init__(self, offset, timestamp, timestamp_type, key, value, crc):
        self._offset = offset
        self._timestamp = timestamp
        self._timestamp_type = timestamp_type
        self._key = key
        self._value = value
        self._crc = crc

    @property
    def offset(self):
        return self._offset

    @property
    def timestamp(self):
        """ Epoch milliseconds
        """
        return self._timestamp

    @property
    def timestamp_type(self):
        """ CREATE_TIME(0) or APPEND_TIME(1)
        """
        return self._timestamp_type

    @property
    def key(self):
        """ Bytes key or None
        """
        return self._key

    @property
    def value(self):
        """ Bytes value or None
        """
        return self._value

    @property
    def headers(self):
        return []

    @property
    def checksum(self):
        return self._crc

    def __repr__(self):
        return f'LegacyRecord(offset={self._offset!r}, timestamp={self._timestamp!r}, timestamp_type={self._timestamp_type!r}, key={self._key!r}, value={self._value!r}, crc={self._crc!r})'