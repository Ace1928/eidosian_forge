import base64
import calendar
import random
from io import BytesIO
from itertools import cycle
from typing import Sequence, Union
from unittest import skipIf
from urllib.parse import clear_cache  # type: ignore[attr-defined]
from urllib.parse import urlparse, urlunsplit
from zope.interface import directlyProvides, providedBy, provider
from zope.interface.verify import verifyObject
import hamcrest
from twisted.internet import address
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.task import Clock
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.protocols import loopback
from twisted.python.compat import iterbytes, networkString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.test.test_internet import DummyProducer
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
from twisted.web import http, http_headers, iweb
from twisted.web.http import PotentialDataLoss, _DataLoss, _IdentityTransferDecoder
from twisted.web.test.requesthelper import (
from ._util import assertIsFilesystemTemporary
def test_transferEncodingIdentity(self):
    """
        A request with a valid C{content-length} and a
        C{transfer-encoding} whose value is C{identity} succeeds.
        """
    body = []

    class SuccessfulRequest(http.Request):
        processed = False

        def process(self):
            body.append(self.content.read())
            self.setHeader(b'content-length', b'0')
            self.finish()
    request = b'GET / HTTP/1.1\nHost: host.invalid\nContent-Length: 2\nTransfer-Encoding: identity\n\nok\n'
    channel = self.runRequest(request, SuccessfulRequest, False)
    self.assertEqual(body, [b'ok'])
    self.assertEqual(channel.transport.value(), b'HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n')