from __future__ import annotations
from typing import Sequence
from twisted.trial.unittest import TestCase
from twisted.web.http_headers import Headers
from twisted.web.test.requesthelper import (
def test_removeHeaderDoesntExist(self) -> None:
    """
        L{Headers.removeHeader} is a no-operation when the specified header is
        not found.
        """
    h = Headers()
    h.removeHeader('test')
    self.assertEqual(list(h.getAllRawHeaders()), [])