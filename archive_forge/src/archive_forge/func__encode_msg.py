import struct
import time
from binascii import crc32
import aiokafka.codec as codecs
from aiokafka.codec import (
from aiokafka.errors import CorruptRecordException, UnsupportedCodecError
from aiokafka.util import NO_EXTENSIONS
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