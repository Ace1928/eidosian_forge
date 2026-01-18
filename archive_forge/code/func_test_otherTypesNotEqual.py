from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_otherTypesNotEqual(self) -> None:
    """
        L{URL} is not equal (C{==}) to other types.
        """
    u = URL.fromText('http://localhost/')
    self.assertFalse(u == 42, 'URL must not equal a number.')
    self.assertFalse(u == object(), 'URL must not equal an object.')
    self.assertNotEqual(u, 42)
    self.assertNotEqual(u, object())