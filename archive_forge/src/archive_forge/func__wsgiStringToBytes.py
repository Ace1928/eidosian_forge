from collections.abc import Sequence
from sys import exc_info
from warnings import warn
from zope.interface import implementer
from twisted.internet.threads import blockingCallFromThread
from twisted.logger import Logger
from twisted.python.failure import Failure
from twisted.web.http import INTERNAL_SERVER_ERROR
from twisted.web.resource import IResource
from twisted.web.server import NOT_DONE_YET
def _wsgiStringToBytes(string):
    """
        Convert C{string} from a WSGI "bytes-as-unicode" string to an
        ISO-8859-1 byte string.

        @type string: C{str}
        @rtype: C{bytes}

        @raise UnicodeEncodeError: If C{string} contains non-ISO-8859-1 chars.
        """
    return string.encode('iso-8859-1')