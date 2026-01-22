from typing import TYPE_CHECKING, Callable, List, Optional
from zope.interface import Attribute, Interface
from twisted.cred.credentials import IUsernameDigestHash
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IPushProducer
from twisted.web.http_headers import Headers
class IResponse(Interface):
    """
    An object representing an HTTP response received from an HTTP server.

    @since: 11.1
    """
    version = Attribute("A three-tuple describing the protocol and protocol version of the response.  The first element is of type L{str}, the second and third are of type L{int}.  For example, C{(b'HTTP', 1, 1)}.")
    code = Attribute('The HTTP status code of this response, as a L{int}.')
    phrase = Attribute('The HTTP reason phrase of this response, as a L{str}.')
    headers = Attribute('The HTTP response L{Headers} of this response.')
    length = Attribute('The L{int} number of bytes expected to be in the body of this response or L{UNKNOWN_LENGTH} if the server did not indicate how many bytes to expect.  For I{HEAD} responses, this will be 0; if the response includes a I{Content-Length} header, it will be available in C{headers}.')
    request = Attribute('The L{IClientRequest} that resulted in this response.')
    previousResponse = Attribute('The previous L{IResponse} from a redirect, or L{None} if there was no previous response. This can be used to walk the response or request history for redirections.')

    def deliverBody(protocol):
        """
        Register an L{IProtocol<twisted.internet.interfaces.IProtocol>} provider
        to receive the response body.

        The protocol will be connected to a transport which provides
        L{IPushProducer}.  The protocol's C{connectionLost} method will be
        called with:

            - ResponseDone, which indicates that all bytes from the response
              have been successfully delivered.

            - PotentialDataLoss, which indicates that it cannot be determined
              if the entire response body has been delivered.  This only occurs
              when making requests to HTTP servers which do not set
              I{Content-Length} or a I{Transfer-Encoding} in the response.

            - ResponseFailed, which indicates that some bytes from the response
              were lost.  The C{reasons} attribute of the exception may provide
              more specific indications as to why.
        """

    def setPreviousResponse(response):
        """
        Set the reference to the previous L{IResponse}.

        The value of the previous response can be read via
        L{IResponse.previousResponse}.
        """