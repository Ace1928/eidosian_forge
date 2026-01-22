import re
from hashlib import md5
from typing import List
from twisted.internet import defer, error, interfaces
from twisted.mail._except import (
from twisted.protocols import basic, policies
from twisted.python import log

        Send a QUIT command to disconnect from the server.

        @rtype: L{Deferred <defer.Deferred>} which successfully fires with
            L{bytes} or fails with L{ServerErrorResponse}
        @return: A deferred which fires when the server response is received.
            On an OK response, the deferred succeeds with the server
            response minus the status indicator.  On an ERR response, the
            deferred fails with a server error response failure.
        