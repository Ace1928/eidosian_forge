import struct
import time
import aiokafka.codec as codecs
from aiokafka.errors import CorruptRecordException, UnsupportedCodecError
from aiokafka.util import NO_EXTENSIONS
from aiokafka.codec import (
from .util import decode_varint, encode_varint, calc_crc32c, size_of_varint
class _DefaultRecordMetadataPy:
    __slots__ = ('_size', '_timestamp', '_offset')

    def __init__(self, offset, size, timestamp):
        self._offset = offset
        self._size = size
        self._timestamp = timestamp

    @property
    def offset(self):
        return self._offset

    @property
    def crc(self):
        return None

    @property
    def size(self):
        return self._size

    @property
    def timestamp(self):
        return self._timestamp

    def __repr__(self):
        return f'DefaultRecordMetadata(offset={self._offset!r}, size={self._size!r}, timestamp={self._timestamp!r})'