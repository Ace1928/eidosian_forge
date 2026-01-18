from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
@staticmethod
def unpack_false(data):
    """This method consumes none of the provided bytes and returns False.

        :type data: bytes
        :param data: The bytes to parse from. This is ignored in this method.

        :rtype: tuple
        :rtype: (bool, int)
        :returns: The tuple (False, 0)
        """
    return (False, 0)