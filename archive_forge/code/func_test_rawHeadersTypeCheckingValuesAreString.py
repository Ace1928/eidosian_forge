from __future__ import annotations
from typing import Sequence
from twisted.trial.unittest import TestCase
from twisted.web.http_headers import Headers
from twisted.web.test.requesthelper import (
def test_rawHeadersTypeCheckingValuesAreString(self) -> None:
    """
        L{Headers.setRawHeaders} requires values to a L{list} of L{bytes} or
        L{str} strings.
        """
    h = Headers()
    e = self.assertRaises(TypeError, h.setRawHeaders, b'key', [b'bar', None])
    self.assertEqual(e.args[0], "Header value at position 1 is an instance of <class 'NoneType'>, not bytes or str")