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
def test_useReactorArgument(self):
    """
        L{twcgi.FilteredScript.runProcess} uses the reactor passed as an
        argument to the constructor.
        """

    class FakeReactor:
        """
            A fake reactor recording whether spawnProcess is called.
            """
        called = False

        def spawnProcess(self, *args, **kwargs):
            """
                Set the C{called} flag to C{True} if C{spawnProcess} is called.

                @param args: Positional arguments.
                @param kwargs: Keyword arguments.
                """
            self.called = True
    fakeReactor = FakeReactor()
    request = DummyRequest(['a', 'b'])
    request.client = address.IPv4Address('TCP', '127.0.0.1', 12345)
    resource = twcgi.FilteredScript('dummy-file', reactor=fakeReactor)
    _render(resource, request)
    self.assertTrue(fakeReactor.called)