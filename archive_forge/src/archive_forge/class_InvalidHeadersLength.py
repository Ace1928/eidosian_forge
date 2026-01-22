from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
class InvalidHeadersLength(ParserError):
    """Headers length is longer than the maximum."""

    def __init__(self, length):
        message = 'Header length of {} exceeded the maximum of {}'.format(length, _MAX_HEADERS_LENGTH)
        super().__init__(message)