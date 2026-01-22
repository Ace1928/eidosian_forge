from typing import TYPE_CHECKING, Callable, List, Optional
from zope.interface import Attribute, Interface
from twisted.cred.credentials import IUsernameDigestHash
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IPushProducer
from twisted.web.http_headers import Headers
class _IRequestEncoder(Interface):
    """
    An object encoding data passed to L{IRequest.write}, for example for
    compression purpose.

    @since: 12.3
    """

    def encode(data):
        """
        Encode the data given and return the result.

        @param data: The content to encode.
        @type data: L{str}

        @return: The encoded data.
        @rtype: L{str}
        """

    def finish():
        """
        Callback called when the request is closing.

        @return: If necessary, the pending data accumulated from previous
            C{encode} calls.
        @rtype: L{str}
        """