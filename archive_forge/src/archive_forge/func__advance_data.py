from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
def _advance_data(self, consumed):
    self._data = self._data[consumed:]