import re
from zope.interface import implementer
from twisted.internet.defer import (
from twisted.internet.error import ConnectionDone
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.logger import Logger
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import networkString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.reflect import fullyQualifiedName
from twisted.web.http import (
from twisted.web.http_headers import Headers
from twisted.web.iweb import UNKNOWN_LENGTH, IClientRequest, IResponse
def _ensureValidMethod(method):
    """
    An HTTP method is an HTTP token, which consists of any visible
    ASCII character that is not a delimiter (i.e. one of
    C{"(),/:;<=>?@[\\]{}}.)

    @param method: the method to check
    @type method: L{bytes}

    @return: the method if it is valid
    @rtype: L{bytes}

    @raise ValueError: if the method is not valid

    @see: U{https://tools.ietf.org/html/rfc7230#section-3.1.1},
        U{https://tools.ietf.org/html/rfc7230#section-3.2.6},
        U{https://tools.ietf.org/html/rfc5234#appendix-B.1}
    """
    if _VALID_METHOD.match(method):
        return method
    raise ValueError(f'Invalid method {method!r}')