from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
def _parse_message_bytes(self):
    message_bytes = self._data[_PRELUDE_LENGTH - 4:self._prelude.payload_end]
    return message_bytes