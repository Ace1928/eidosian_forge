from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
@staticmethod
def unpack_utf8_string(data, length_byte_size=2):
    """Parse a variable length utf-8 string from the bytes.

        The bytes are expected to be in the following format:
            [ length ][0 ... length bytes]
        where length is an unsigned integer represented in the smallest number
        of bytes to hold the maximum length of the array and the following
        bytes are a valid utf-8 string.

        :type data: bytes
        :param bytes: The bytes to parse from.

        :type length_byte_size: int
        :param length_byte_size: The byte size of the preceding integer that
        represents the length of the array. Supported values are 1, 2, and 4.

        :rtype: (str, int)
        :returns: A tuple containing the (utf-8 string, bytes consumed).
        """
    array_bytes, consumed = DecodeUtils.unpack_byte_array(data, length_byte_size)
    return (array_bytes.decode('utf-8'), consumed)