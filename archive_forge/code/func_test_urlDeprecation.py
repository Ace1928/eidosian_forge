from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_urlDeprecation(self) -> None:
    """
        L{twisted.python.url} is deprecated since Twisted 17.5.0.
        """
    from twisted.python import url
    url
    warningsShown = self.flushWarnings([self.test_urlDeprecation])
    self.assertEqual(1, len(warningsShown))
    self.assertEqual('twisted.python.url was deprecated in Twisted 17.5.0: Please use hyperlink from PyPI instead.', warningsShown[0]['message'])