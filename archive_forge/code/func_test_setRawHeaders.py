from __future__ import annotations
from typing import Sequence
from twisted.trial.unittest import TestCase
from twisted.web.http_headers import Headers
from twisted.web.test.requesthelper import (
def test_setRawHeaders(self) -> None:
    """
        L{Headers.setRawHeaders} accepts mixed L{str} and L{bytes}.
        """
    h = Headers()
    h.setRawHeaders(b'bytes', [b'bytes'])
    h.setRawHeaders('str', ['str'])
    h.setRawHeaders('mixed-str', [b'bytes', 'str'])
    h.setRawHeaders(b'mixed-bytes', ['str', b'bytes'])
    self.assertEqual(h.getRawHeaders(b'Bytes'), [b'bytes'])
    self.assertEqual(h.getRawHeaders('Str'), ['str'])
    self.assertEqual(h.getRawHeaders('Mixed-Str'), ['bytes', 'str'])
    self.assertEqual(h.getRawHeaders(b'Mixed-Bytes'), [b'str', b'bytes'])