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
def test_pathInfo(self):
    """
        L{twcgi.CGIScript.render} sets the process environment
        I{PATH_INFO} from the request path.
        """

    class FakeReactor:
        """
            A fake reactor recording the environment passed to spawnProcess.
            """

        def spawnProcess(self, process, filename, args, env, wdir):
            """
                Store the C{env} L{dict} to an instance attribute.

                @param process: Ignored
                @param filename: Ignored
                @param args: Ignored
                @param env: The environment L{dict} which will be stored
                @param wdir: Ignored
                """
            self.process_env = env
    _reactor = FakeReactor()
    resource = twcgi.CGIScript(self.mktemp(), reactor=_reactor)
    request = DummyRequest(['a', 'b'])
    request.client = address.IPv4Address('TCP', '127.0.0.1', 12345)
    _render(resource, request)
    self.assertEqual(_reactor.process_env['PATH_INFO'], '/a/b')