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
def test_shortTrailerHeader(self):
    """
        L{_ChunkedTransferDecoder.dataReceived} decodes chunks of input with
        tailer header broken up and delivered in multiple calls.
        """
    L = []
    finished = []
    p = http._ChunkedTransferDecoder(L.append, finished.append)
    for s in iterbytes(b'3\r\nabc\r\n5\r\n12345\r\n0\r\nServer-Timing: total;dur=123.4\r\n\r\n'):
        p.dataReceived(s)
    self.assertEqual(L, [b'a', b'b', b'c', b'1', b'2', b'3', b'4', b'5'])
    self.assertEqual(finished, [b''])
    self.assertEqual(p._trailerHeaders, [b'Server-Timing: total;dur=123.4'])