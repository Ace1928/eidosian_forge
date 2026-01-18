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
def test_urlParameters(self):
    """
        If the CGI script is passed URL parameters, do not fall over,
        as per ticket 9887.
        """
    cgiFilename = self.writeCGI(URL_PARAMETER_CGI)
    portnum = self.startServer(cgiFilename)
    url = b'http://localhost:%d/cgi?param=1234' % (portnum,)
    agent = client.Agent(reactor)
    d = agent.request(b'GET', url)
    d.addCallback(client.readBody)
    d.addCallback(self._test_urlParameters_1)
    return d