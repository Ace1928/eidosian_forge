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
class QueryArgumentsTests(unittest.TestCase):

    def test_urlparse(self):
        """
        For a given URL, L{http.urlparse} should behave the same as L{urlparse},
        except it should always return C{bytes}, never text.
        """

        def urls():
            for scheme in (b'http', b'https'):
                for host in (b'example.com',):
                    for port in (None, 100):
                        for path in (b'', b'path'):
                            if port is not None:
                                host = host + b':' + networkString(str(port))
                                yield urlunsplit((scheme, host, path, b'', b''))

        def assertSameParsing(url, decode):
            """
            Verify that C{url} is parsed into the same objects by both
            L{http.urlparse} and L{urlparse}.
            """
            urlToStandardImplementation = url
            if decode:
                urlToStandardImplementation = url.decode('ascii')
            standardResult = urlparse(urlToStandardImplementation)
            if isinstance(standardResult.scheme, str):
                expected = (standardResult.scheme.encode('utf-8'), standardResult.netloc.encode('utf-8'), standardResult.path.encode('utf-8'), standardResult.params.encode('utf-8'), standardResult.query.encode('utf-8'), standardResult.fragment.encode('utf-8'))
            else:
                expected = (standardResult.scheme, standardResult.netloc, standardResult.path, standardResult.params, standardResult.query, standardResult.fragment)
            scheme, netloc, path, params, query, fragment = http.urlparse(url)
            self.assertEqual((scheme, netloc, path, params, query, fragment), expected)
            self.assertIsInstance(scheme, bytes)
            self.assertIsInstance(netloc, bytes)
            self.assertIsInstance(path, bytes)
            self.assertIsInstance(params, bytes)
            self.assertIsInstance(query, bytes)
            self.assertIsInstance(fragment, bytes)
        clear_cache()
        for url in urls():
            assertSameParsing(url, True)
            assertSameParsing(url, False)
        clear_cache()
        for url in urls():
            assertSameParsing(url, False)
            assertSameParsing(url, True)
        for url in urls():
            clear_cache()
            assertSameParsing(url, True)
            clear_cache()
            assertSameParsing(url, False)

    def test_urlparseRejectsUnicode(self):
        """
        L{http.urlparse} should reject unicode input early.
        """
        self.assertRaises(TypeError, http.urlparse, 'http://example.org/path')