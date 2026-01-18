from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
@staticmethod
def unpack_uint32(data):
    """Parse an unsigned 32-bit integer from the bytes.

        :type data: bytes
        :param data: The bytes to parse from.

        :rtype: (int, int)
        :returns: A tuple containing the (parsed integer value, bytes consumed)
        """
    value = unpack(DecodeUtils.UINT32_BYTE_FORMAT, data[:4])[0]
    return (value, 4)