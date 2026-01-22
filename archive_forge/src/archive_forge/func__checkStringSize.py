import math
import re
from io import BytesIO
from struct import calcsize, pack, unpack
from zope.interface import implementer
from twisted.internet import defer, interfaces, protocol
from twisted.python import log
def _checkStringSize(self, lengthAsString):
    """
        Checks the sanity of lengthAsString.

        Checks if the size of the length specification exceeds the
        size of the string representing self.MAX_LENGTH. If this is
        not the case, the number represented by lengthAsString is
        certainly bigger than self.MAX_LENGTH, and a
        NetstringParseError can be raised.

        This method should make sure that netstrings with extremely
        long length specifications are refused before even attempting
        to convert them to an integer (which might trigger a
        MemoryError).
        """
    if len(lengthAsString) > self._maxLengthSize():
        raise NetstringParseError(self._TOO_LONG % (self.MAX_LENGTH,))