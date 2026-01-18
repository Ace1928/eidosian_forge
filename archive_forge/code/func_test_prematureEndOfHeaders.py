import json
import os
import sys
from io import BytesIO
from twisted.internet import address, error, interfaces, reactor
from twisted.internet.error import ConnectionLost
from twisted.python import failure, log, util
from twisted.trial import unittest
from twisted.web import client, http, http_headers, resource, server, twcgi
from twisted.web.http import INTERNAL_SERVER_ERROR, NOT_FOUND
from twisted.web.test._util import _render
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
import os, sys
import sys
import json
import os
import os
def test_prematureEndOfHeaders(self):
    """
        If the process communicating with L{CGIProcessProtocol} ends before
        finishing writing out headers, the response has I{INTERNAL SERVER
        ERROR} as its status code.
        """
    request = DummyRequest([''])
    protocol = twcgi.CGIProcessProtocol(request)
    protocol.processEnded(failure.Failure(error.ProcessTerminated()))
    self.assertEqual(request.responseCode, INTERNAL_SERVER_ERROR)