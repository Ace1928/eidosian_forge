from twisted.trial.unittest import TestCase
from twisted.web.error import UnsupportedMethod
from twisted.web.http_headers import Headers
from twisted.web.iweb import IRequest
from twisted.web.resource import (
from twisted.web.test.requesthelper import DummyRequest
class BytesReturnedRenderable(Resource):
    """
    A L{Resource} with minimal capabilities to render a response.
    """

    def __init__(self, response: bytes) -> None:
        """
        @param response: A C{bytes} object giving the value to return from
            C{render_GET}.
        """
        Resource.__init__(self)
        self._response = response

    def render_GET(self, request: object) -> bytes:
        """
        Render a response to a I{GET} request by returning a short byte string
        to be written by the server.
        """
        return self._response