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
def _sendResponseHeaders(self):
    """
        Set the response code and response headers on the request object, but
        do not flush them.  The caller is responsible for doing a write in
        order for anything to actually be written out in response to the
        request.

        This must be called in the I/O thread.
        """
    code, message = self.status.split(None, 1)
    code = int(code)
    self.request.setResponseCode(code, _wsgiStringToBytes(message))
    for name, value in self.headers:
        if name.lower() not in ('server', 'date'):
            self.request.responseHeaders.addRawHeader(_wsgiStringToBytes(name), _wsgiStringToBytes(value))