from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
def _prepare_for_next_message(self):
    self._data = self._data[self._prelude.total_length:]
    self._prelude = None