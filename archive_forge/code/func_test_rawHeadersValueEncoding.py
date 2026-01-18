from __future__ import annotations
from typing import Sequence
from twisted.trial.unittest import TestCase
from twisted.web.http_headers import Headers
from twisted.web.test.requesthelper import (
def test_rawHeadersValueEncoding(self) -> None:
    """
        Passing L{str} to L{Headers.setRawHeaders} will encode the name as
        ISO-8859-1 and values as UTF-8.
        """
    h = Headers()
    h.setRawHeaders('á', ['☃', b'foo'])
    self.assertTrue(h.hasHeader(b'\xe1'))
    self.assertEqual(h.getRawHeaders(b'\xe1'), [b'\xe2\x98\x83', b'foo'])