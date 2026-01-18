from __future__ import annotations
from typing import Sequence
from twisted.trial.unittest import TestCase
from twisted.web.http_headers import Headers
from twisted.web.test.requesthelper import (
def test_sanitizeLinearWhitespace(self) -> None:
    """
        Linear whitespace in header names or values is replaced with a
        single space.
        """
    assertSanitized(self, textLinearWhitespaceComponents, sanitizedBytes)