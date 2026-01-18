from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
@staticmethod
def unpack_true(data):
    """This method consumes none of the provided bytes and returns True.

        :type data: bytes
        :param data: The bytes to parse from. This is ignored in this method.

        :rtype: tuple
        :rtype: (bool, int)
        :returns: The tuple (True, 0)
        """
    return (True, 0)