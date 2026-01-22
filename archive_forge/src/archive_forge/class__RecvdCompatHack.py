import math
import re
from io import BytesIO
from struct import calcsize, pack, unpack
from zope.interface import implementer
from twisted.internet import defer, interfaces, protocol
from twisted.python import log
class _RecvdCompatHack:
    """
    Emulates the to-be-deprecated C{IntNStringReceiver.recvd} attribute.

    The C{recvd} attribute was where the working buffer for buffering and
    parsing netstrings was kept.  It was updated each time new data arrived and
    each time some of that data was parsed and delivered to application code.
    The piecemeal updates to its string value were expensive and have been
    removed from C{IntNStringReceiver} in the normal case.  However, for
    applications directly reading this attribute, this descriptor restores that
    behavior.  It only copies the working buffer when necessary (ie, when
    accessed).  This avoids the cost for applications not using the data.

    This is a custom descriptor rather than a property, because we still need
    the default __set__ behavior in both new-style and old-style subclasses.
    """

    def __get__(self, oself, type=None):
        return oself._unprocessed[oself._compatibilityOffset:]