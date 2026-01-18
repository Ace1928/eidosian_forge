from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_similarEqual(self) -> None:
    """
        URLs with equivalent components should compare equal.
        """
    u1 = URL.fromText('http://localhost/')
    u2 = URL.fromText('http://localhost/')
    self.assertEqual(u1, u2)