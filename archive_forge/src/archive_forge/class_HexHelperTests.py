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
class HexHelperTests(unittest.SynchronousTestCase):
    """
    Test the L{http._hexint} and L{http._ishexdigits} helper functions.
    """
    badStrings = (b'', b'0x1234', b'feds', b'-123+123')

    def test_isHex(self):
        """
        L{_ishexdigits()} returns L{True} for nonempy bytestrings containing
        hexadecimal digits.
        """
        for s in (b'10', b'abcdef', b'AB1234', b'fed', b'123467890'):
            self.assertIs(True, http._ishexdigits(s))

    def test_decodes(self):
        """
        L{_hexint()} returns the integer equivalent of the input.
        """
        self.assertEqual(10, http._hexint(b'a'))
        self.assertEqual(16, http._hexint(b'10'))
        self.assertEqual(180146467, http._hexint(b'abCD123'))

    def test_isNotHex(self):
        """
        L{_ishexdigits()} returns L{False} for bytestrings that don't contain
        hexadecimal digits, including the empty string.
        """
        for s in self.badStrings:
            self.assertIs(False, http._ishexdigits(s))

    def test_decodeNotHex(self):
        """
        L{_hexint()} raises L{ValueError} for bytestrings that can't
        be decoded.
        """
        for s in self.badStrings:
            self.assertRaises(ValueError, http._hexint, s)