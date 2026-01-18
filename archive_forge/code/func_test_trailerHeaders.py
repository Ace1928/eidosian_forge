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
def test_trailerHeaders(self):
    """
        L{_ChunkedTransferDecoder.dataReceived} decodes chunked-encoded data
        and ignores trailer headers which come after the terminating zero-length
        chunk.
        """
    L = []
    finished = []
    p = http._ChunkedTransferDecoder(L.append, finished.append)
    p.dataReceived(b'3\r\nabc\r\n5\r\n12345\r\n')
    p.dataReceived(b'a\r\n0123456789\r\n0\r\nServer-Timing: total;dur=123.4\r\nExpires: Wed, 21 Oct 2015 07:28:00 GMT\r\n\r\n')
    self.assertEqual(L, [b'abc', b'12345', b'0123456789'])
    self.assertEqual(finished, [b''])
    self.assertEqual(p._trailerHeaders, [b'Server-Timing: total;dur=123.4', b'Expires: Wed, 21 Oct 2015 07:28:00 GMT'])