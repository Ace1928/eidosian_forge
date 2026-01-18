from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
@staticmethod
def unpack_uuid(data):
    """Parse a 16-byte uuid from the bytes.

        :type data: bytes
        :param data: The bytes to parse from.

        :rtype: (bytes, int)
        :returns: A tuple containing the (uuid bytes, bytes consumed).
        """
    return (data[:16], 16)