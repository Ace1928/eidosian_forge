from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_queryRemove(self) -> None:
    """
        L{URL.remove} removes all instances of a query parameter.
        """
    url = URL.fromText('https://example.com/a/b/?foo=1&bar=2&foo=3')
    self.assertEqual(url.remove('foo'), URL.fromText('https://example.com/a/b/?bar=2'))