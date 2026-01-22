import math
import re
from io import BytesIO
from struct import calcsize, pack, unpack
from zope.interface import implementer
from twisted.internet import defer, interfaces, protocol
from twisted.python import log
class StringTooLongError(AssertionError):
    """
    Raised when trying to send a string too long for a length prefixed
    protocol.
    """