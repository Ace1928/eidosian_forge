from __future__ import annotations
from typing import Sequence
from twisted.trial.unittest import TestCase
from twisted.web.http_headers import Headers
from twisted.web.test.requesthelper import (
def test_otherComparison(self) -> None:
    """
        An instance of L{Headers} does not compare equal to other unrelated
        objects.
        """
    h = Headers()
    self.assertNotEqual(h, ())
    self.assertNotEqual(h, object())
    self.assertNotEqual(h, 'foo')